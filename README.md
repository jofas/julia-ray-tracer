# Julia set ray tracing shader

Mostly a portation of Keenan Crane's implementation from
his paper [Ray Tracing Quaternion Julia Sets on the GPU](https://www.cs.cmu.edu/~kmcrane/Projects/QuaternionJulia/paper.pdf) to [Unity](https://unity3d.com) and directly to WebGL2 + WebAssembly.

Also useful: [Quaternion Julia Fractals](http://paulbourke.net/fractals/quatjulia/) by Paul Bourke.

## WebGL2 + WebAssembly

In case the Browser/Device does not have support for WebGL2 or WebAssembly the program falls back to rendering a l-system to a svg.
