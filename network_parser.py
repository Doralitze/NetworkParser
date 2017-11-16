#! /usr/bin/env python3

"""
BSD 3-Clause License

Copyright (c) 2017, Doralitze
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from vispy import gloo
from vispy import app
from vispy import io
from vispy.util.ptime import time
from vispy.gloo.util import _screenshot
from vispy.util.transforms import perspective, translate, rotate
import matplotlib.pyplot as matplt
import csv
import argparse
import numpy
import math
import threading
import os

# Define required classes
class Node(object):
    """A class to represent a network node."""

    position = None
    connections = []
    name: str = None

    def __init__(self, pos_x: float, pos_y: float, pos_z: float, name: str,
        connections = []):
        super(Node, self).__init__()
        self.position = numpy.array([pos_x, pos_y, pos_z])
        self.name = name
        self.connections = []
        for c in connections:
            self.connections.append(c)

    def __str__(self):
        s = "node: {name: " + self.name + ", position: ["
        s += str(self.position) + "], connections: "
        s += str(self.connections) + "}"
        return s

class Canvas(app.Canvas):

    shader_nodes: gloo.Program
    shader_traces: gloo.Program

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))
        ps = self.pixel_scale

        # Create vertices
        n = len(nodes)
        b_lengh, trace_data = compute_connections()
        data = numpy.zeros(n, [('a_position', numpy.float32, 3),
                            ('a_bg_color', numpy.float32, 4),
                            ('a_fg_color', numpy.float32, 4),
                            ('a_size', numpy.float32, 1)])

        for i in range(0, len(nodes)):
            na: Node = nodes[i]
            data["a_position"][i] = na.position
            if args.verbosity > 2:
                print("Added node " + str(i) + " to GPU buffer.")
        if args.verbosity > 3:
            print("Using the following object buffer on GPU:")
            print(str(data["a_position"]))
        data['a_bg_color'] = numpy.random.uniform(0.85, 1.00, (n, 4))
        data['a_fg_color'] = 0, 0, 0, 1
        data['a_size'] = numpy.random.uniform(5*ps, 10*ps, n)
        u_linewidth = 1.0
        u_antialias = 1.0

        self.translate = 5
        self.shader_nodes = gloo.Program(vert_nodes, frag_nodes)
        self.shader_traces = gloo.Program(vert_traces, frag_traces)
        self.program = self.shader_nodes
        self.view = translate((0, 0, -self.translate))
        self.model = numpy.eye(4, dtype=numpy.float32)
        self.projection = numpy.eye(4, dtype=numpy.float32)

        self.apply_zoom()

        self.program.bind(gloo.VertexBuffer(data))
        self.shader_traces.bind(gloo.VertexBuffer(trace_data))
        self.program['u_linewidth'] = u_linewidth
        self.program['u_antialias'] = u_antialias
        self.shader_traces['u_linewidth'] = u_linewidth
        self.shader_traces['u_antialias'] = u_antialias
        self.program['u_render_selection'] = float(args.rendermethod)
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.shader_traces['u_model'] = self.model
        self.shader_traces['u_view'] = self.view
        self.program['u_size'] = 5 / self.translate
        self.shader_traces['u_size'] = 5 / self.translate
        self.shader_traces['u_r'] = 1 - args.thickness

        self.theta = 0
        self.phi = 0

        gloo.set_state('translucent', clear_color='white')

        self.timer = app.Timer('auto', connect=self.on_timer, start=True)

        self.show()

    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()
        if event.text == 'w':
            self.theta += 2
            self.apply_rotation()
        if event.text == 's':
            self.theta -= 2
            self.apply_rotation()
        if event.text == 'd':
            self.phi += 2
            self.apply_rotation()
        if event.text == 'a':
            self.phi -= 2
            self.apply_rotation()

    def apply_rotation(self):
        self.model = numpy.dot(rotate(self.theta, (0, 0, 1)),
                            rotate(self.phi, (0, 1, 0)))
        self.shader_nodes['u_model'] = self.model
        self.shader_traces['u_model'] = self.model
        self.update()

    def on_timer(self, event):
        self.theta += .5
        self.phi += .5
        self.apply_rotation()

    def on_resize(self, event):
        self.apply_zoom()

    def on_mouse_wheel(self, event):
        self.translate -= event.delta[1]
        self.translate = max(2, self.translate)
        self.view = translate((0, 0, -self.translate))

        self.shader_nodes['u_view'] = self.view
        self.shader_nodes['u_size'] = 5 / self.translate
        self.shader_traces['u_view'] = self.view
        self.shader_traces['u_size'] = 5 / self.translate
        self.update()

    def on_draw(self, event):
        if args.outputfile:
            faces = [ [0, 0], [90, 0], [180, 0], [270, 0], [0, 90], [0, 270] ]
            self.model = numpy.dot(rotate(faces[args.faces - 1][0], (0, 0, 1)),
                                rotate(faces[args.faces - 1][0], (0, 1, 0)))
            self.shader_nodes['u_model'] = self.model
            self.shader_traces['u_model'] = self.model
        gloo.clear()
        self.program = self.shader_nodes
        self.program.draw('points')
        self.program = self.shader_traces
        self.program.draw('lines')
        if args.outputfile:
            self.im = _screenshot((0, 0, self.size[0], self.size[1]))
            app.quit()


    def apply_zoom(self):
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)
        self.shader_nodes['u_projection'] = self.projection
        self.shader_traces['u_projection'] = self.projection

# define required shaders
vert_nodes = """
#version 120
// Uniforms
// ------------------------------------
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_linewidth;
uniform float u_antialias;
uniform float u_render_selection;
uniform float u_size;
// Attributes
// ------------------------------------
attribute vec3  a_position; //Position of each node
attribute vec4  a_fg_color;
attribute vec4  a_bg_color;
attribute float a_size;
// Varyings
// ------------------------------------
varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_size;
varying float v_linewidth;
varying float v_antialias;
varying float v_render_selection;
void main (void) {
    v_size = a_size * u_size;
    v_linewidth = u_linewidth;
    v_antialias = u_antialias;
    v_render_selection = u_render_selection;
    v_fg_color  = a_fg_color;
    v_bg_color  = a_bg_color;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    gl_PointSize = v_size + 2*(v_linewidth + 1.5*v_antialias);
}
"""

frag_nodes = """
#version 120
// Constants
// ------------------------------------
// Varyings
// ------------------------------------
varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_size;
varying float v_linewidth;
varying float v_antialias;
varying float v_render_selection;

// Functions
// ------------------------------------
// ----------------
float disc(vec2 P, float size)
{
    float r = length((P.xy - vec2(0.5,0.5))*size);
    r -= v_size/2;
    return r;
}
// ----------------
float arrow_right(vec2 P, float size)
{
    float r1 = abs(P.x -.50)*size + abs(P.y -.5)*size - v_size/2;
    float r2 = abs(P.x -.25)*size + abs(P.y -.5)*size - v_size/2;
    float r = max(r1,-r2);
    return r;
}
// ----------------
float ring(vec2 P, float size)
{
    float r1 = length((gl_PointCoord.xy - vec2(0.5,0.5))*size) - v_size/2;
    float r2 = length((gl_PointCoord.xy - vec2(0.5,0.5))*size) - v_size/4;
    float r = max(r1,-r2);
    return r;
}
// ----------------
float clober(vec2 P, float size)
{
    const float PI = 3.14159265358979323846264;
    const float t1 = -PI/2;
    const vec2  c1 = 0.2*vec2(cos(t1),sin(t1));
    const float t2 = t1+2*PI/3;
    const vec2  c2 = 0.2*vec2(cos(t2),sin(t2));
    const float t3 = t2+2*PI/3;
    const vec2  c3 = 0.2*vec2(cos(t3),sin(t3));
    float r1 = length((gl_PointCoord.xy- vec2(0.5,0.5) - c1)*size);
    r1 -= v_size/3;
    float r2 = length((gl_PointCoord.xy- vec2(0.5,0.5) - c2)*size);
    r2 -= v_size/3;
    float r3 = length((gl_PointCoord.xy- vec2(0.5,0.5) - c3)*size);
    r3 -= v_size/3;
    float r = min(min(r1,r2),r3);
    return r;
}
// ----------------
float square(vec2 P, float size)
{
    float r = max(abs(gl_PointCoord.x -.5)*size,
                  abs(gl_PointCoord.y -.5)*size);
    r -= v_size/2;
    return r;
}
// ----------------
float diamond(vec2 P, float size)
{
    float r = abs(gl_PointCoord.x -.5)*size + abs(gl_PointCoord.y -.5)*size;
    r -= v_size/2;
    return r;
}
// ----------------
float vbar(vec2 P, float size)
{
    float r1 = max(abs(gl_PointCoord.x -.75)*size,
                   abs(gl_PointCoord.x -.25)*size);
    float r3 = max(abs(gl_PointCoord.x -.5)*size,
                   abs(gl_PointCoord.y -.5)*size);
    float r = max(r1,r3);
    r -= v_size/2;
    return r;
}
// ----------------
float hbar(vec2 P, float size)
{
    float r2 = max(abs(gl_PointCoord.y -.75)*size,
                   abs(gl_PointCoord.y -.25)*size);
    float r3 = max(abs(gl_PointCoord.x -.5)*size,
                   abs(gl_PointCoord.y -.5)*size);
    float r = max(r2,r3);
    r -= v_size/2;
    return r;
}
// ----------------
float cross(vec2 P, float size)
{
    float r1 = max(abs(gl_PointCoord.x -.75)*size,
                   abs(gl_PointCoord.x -.25)*size);
    float r2 = max(abs(gl_PointCoord.y -.75)*size,
                   abs(gl_PointCoord.y -.25)*size);
    float r3 = max(abs(gl_PointCoord.x -.5)*size,
                   abs(gl_PointCoord.y -.5)*size);
    float r = max(min(r1,r2),r3);
    r -= v_size/2;
    return r;
}
// Main
// ------------------------------------
void main()
{
    float size = v_size +2*(v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;
    float r;

    if(v_render_selection == 0.0f)
        r = disc(gl_PointCoord, size);
    else if(v_render_selection == 1.0f)
        r = square(gl_PointCoord, size);
    else if(v_render_selection == 2.0f)
        r = ring(gl_PointCoord, size);
    else if(v_render_selection == 3.0f)
        r = arrow_right(gl_PointCoord, size);
    else if(v_render_selection == 4.0f)
        r = diamond(gl_PointCoord, size);
    else if(v_render_selection == 5.0f)
        r = cross(gl_PointCoord, size);
    else if(v_render_selection == 6.0f)
        r = clober(gl_PointCoord, size);
    else if(v_render_selection == 7.0f)
        r = hbar(gl_PointCoord, size);
    else
        r = vbar(gl_PointCoord, size);



    float d = abs(r) - t;
    if( r > (v_linewidth/2.0+v_antialias))
    {
        discard;
    }
    else if( d < 0.0 )
    {
       gl_FragColor = v_fg_color;
    }
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (r > 0)
            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        else
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
    }
}
"""

vert_traces = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_linewidth;
uniform float u_antialias;
uniform float u_size;
uniform float u_r;

attribute vec3 a_position;

varying float v_linewidth;
varying float v_antialias;
varying float v_size;
varying float v_r;

void main (void) {
    v_linewidth = u_linewidth;
    v_antialias = u_antialias;
    v_size = u_size;
    v_r = u_r;
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    gl_PointSize = v_size + 2*(v_linewidth + 1.5*v_antialias);
}
"""

frag_traces = """

varying float v_linewidth;
varying float v_antialias;
varying float v_r;

void main (void) {
    float t = v_linewidth/2.0-v_antialias;
    float d = abs(v_r) - t;
    if( d < 0.0 )
    {
       gl_FragColor = vec4(1,0,0,1);
    }
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (v_r > 0)
            gl_FragColor = vec4(vec4(1,0,0,1).rgb, alpha*vec4(0.9).a);
        else
            gl_FragColor = mix(vec4(1,1,1,1), vec4(1,0,0,1), alpha);
    }
}
"""

#declaring required functions
def parse_node(row):
    name: str = row[0]
    px: float = float(row[1])
    py: float = float(row[2])
    pz: float = float(row[3])
    c = []
    for i in range(4, len(row)):
        try:
            c.append(row[i])
        except Exception as e:
            if args.verbosity > 2:
                print("The following exception was thrown doing the parsing " +
                "of the row:\n" + str(e))
    return Node(px, py, pz, name, c)


def find_node(name):
    """Find the correct node using its name"""
    for n in nodes:
        if n.name == name:
            return n


def check_network_integrity(start: int, stop: int):
    """Check if the network is correctly within the given intervall"""
    if args.verbosity > 2:
        print("Checking node list from " + str(start) + " to " + str(stop))
    for i in range(start, stop + 1):
        if i >= len(nodes):
            if args.verbosity > 3:
                print("Skipping from " + str(i) + " due to IOoB")
            return True
        n = nodes[i]
        if args.verbosity > 3:
            print("Checking element: " + str(i) + " -> " + str(n))
        for c in n.connections:
            if not find_node(c):
                if not args.quiet:
                    print("There are node connections referencing ghosts.")
                    print("Check your file.")
                exit(1)
    return True


def compute_connections():
    c = []
    def is_present(con):
        for k in c:
            if ((k[0] == con[0] and k[1] == con[1]) or
                (k[0] == con[1] and k[1] == con[0])):
                return True

    for n in nodes:
        for t in n.connections:
            a = find_node(t)
            if not is_present((n,a)):
                c.append((n,a))
    trace_data = numpy.zeros(len(c) * 2, [('a_position', numpy.float32, 3)])
    for i in range(0, len(c)):
        if args.verbosity > 2:
            print("adding connection [" + str(i) + "]: " + str(c[i]))
        trace_data['a_position'][i * 2][0]       = c[i][1].position[0]
        trace_data['a_position'][i * 2][1]       = c[i][1].position[1]
        trace_data['a_position'][i * 2][2]       = c[i][1].position[2]
        trace_data['a_position'][(i * 2) + 1][0] = c[i][0].position[0]
        trace_data['a_position'][(i * 2) + 1][1] = c[i][0].position[1]
        trace_data['a_position'][(i * 2) + 1][2] = c[i][0].position[2]
    return len(c), trace_data


# declaring global variables
nodes = []
global_node_maximum = 0.0

# Parse command line arguments
args = None
if True:  # Just for the sake of having the parser not globally aviable
    def restricted_float(x):
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
        return x
    parser = argparse.ArgumentParser(description="Render a network of nodes " +
                                    "specified by a csv file")
    parser.add_argument("inputfile", help="specify the csv file to parse",
           type=str)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-v", "--verbosity",
                 help="increase output verbosity", action="count", default=0)
    output_group.add_argument("--quiet", action="store_true",
                 help="Disable entire output of programm")
    parser.add_argument("-f", "--faces", default=1, help="Define which side " +
                    "should be rendered", type=int, choices=[1,2,3,4,5,6])
    parser.add_argument("-d", "--delimiter", type=str, default=";",
           help="Specify the delimiter to use in order to parse the csv file")
    parser.add_argument("-q", "--quotes", type=str, default="|",
           help="Specify the quote char to use in order to parse the csv file")
    parser.add_argument("-e", "--encoding", type=str, default="utf-8",
                    help="Specify the csv files encoding")
    parser.add_argument("-rm", "--rendermethod", type=int, choices=[0,1,2,3,4,
            5,6,7,8], help="Specify how to render the nodes.", default=0)
    parser.add_argument("-o", "--outputfile", type=str,
            help="Specify an output file")
    parser.add_argument("--resolution", type=int, default=250,
            help="Specify the resolution in dpi")
    parser.add_argument("--thickness", type=restricted_float, default=0.8,
            help="Specify the line thickness in [0 .. 1]")
    args = parser.parse_args()

# Parse given csv file
with open(args.inputfile, newline='', encoding=args.encoding) as csvfile:
    rowreader = csv.reader(csvfile, delimiter=args.delimiter,
                quotechar=args.quotes)
    for r in rowreader:
        if args.verbosity > 3:
            print("Parsing row: " + str(r))
            print("Current maximum: " + str(global_node_maximum))
        n = parse_node(r)
        if args.verbosity > 1:
            print("Appending the following " + str(n))
        nodes.append(n)

# Validate the network
if len(nodes) > 0:
    size = len(nodes)
    stepsize = math.floor(size / os.cpu_count())
    if size < os.cpu_count():
        stepsize = size
    if args.verbosity > 1:
        print("Checking a " + str(size) + " long node list using a step of " +
              str(stepsize))
    pointer = -1
    while pointer < size:
        pointer += 1
        threading.Thread(target=check_network_integrity,
                         args=(pointer, pointer + stepsize)).start()
        pointer += stepsize
else:
    if args.verbosity > 0:
        print("Exiting due to empty node list")
    exit()

# normalize the node positions
for n in nodes:
    px = n.position[0]
    py = n.position[1]
    pz = n.position[2]
    if px > global_node_maximum:
        global_node_maximum = px
    if py > global_node_maximum:
        global_node_maximum = py
    if py > global_node_maximum:
        global_node_maximum = py
for n in nodes:
    for i in range(0,3):
        n.position[i] = n.position[i] / global_node_maximum

# render the network
if __name__ == '__main__':
    c = Canvas()
    app.run()
    if args.outputfile:
        size = c.size
        render = c.im
        if not args.quiet:
            # print("Finished rendering of data (%.1f ms of GPU time)." % \
            # (c._time*1e3))
            pass

        matplt.figure(figsize=(size[0]/100., size[1]/100.), dpi=args.resolution)
        matplt.xlabel("X-Dimension")
        matplt.ylabel("Y-Dimension")
        matplt.imshow(render, interpolation='none')
        #matplt.show()
        matplt.savefig(args.outputfile, dpi=args.resolution)
