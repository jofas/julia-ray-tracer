#[macro_use]
extern crate stdweb;
#[macro_use]
extern crate stdweb_derive;
#[macro_use]
extern crate serde_derive;

extern crate quaternion;

use stdweb::web::{ document
                 , window
                 , TypedArray
                 , IParentNode };
use stdweb::web::html_element::CanvasElement;
use stdweb::unstable::{TryInto, TryFrom};

use quaternion as q;

use std::ops::{Add, Sub, Mul};
use std::convert::{From, Into};
use std::f32::consts::PI;
use std::rc::Rc;
use std::cell::RefCell;

mod webgl2 {
  #![allow( dead_code, unused_parens, unused_imports
          , non_camel_case_types )]
  include!("webgl2_context.rs");
}

use webgl2::{ WebGL2RenderingContext as gl
            , WebGLUniformLocation };

// Camera radius
static CR: f32 = 6.;

static _BOUNDINGSPHERERADIUS: f32 = 5.0;
static _THRESHOLD: f32 = 4.0;
static _EPSILON: f32 = 1e-3;

static _MAXITER: i32 = 8;
static _NORMALITER: i32 = 8;

// VERT_SRC {{{
static VERT_SRC: &str = "#version 300 es
  in vec3 a_position;
  out vec3 world_pos;

  uniform vec3 port0;
  uniform vec3 port1;
  uniform vec3 port2;
  uniform vec3 port3;

  void main() {

    if (a_position == port0) {
      gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
    } else if (a_position == port1) {
      gl_Position = vec4(-1.0, 1.0, 0.0, 1.0);
    } else if (a_position == port2) {
      gl_Position = vec4(1.0, -1.0, 0.0, 1.0);
    } else if (a_position == port3) {
      gl_Position = vec4(1.0, 1.0, 0.0, 1.0);
    }

    world_pos = 3.0 * a_position;
    //gl_Position = vec4(a_position.x, a_position.y, 0.0, 1.0);
  }";
// }}}

// FRAG_SRC {{{
fn frag_src() -> String {
  format!("#version 300 es

    // Based on Keenan Crane's implementation:
    // www.cs.cmu.edu/~kmcrane/Projects/QuaternionJulia/
    // paper.pdf

    precision highp float;

    in vec3 world_pos;
    out vec4 outColor;

    uniform vec4 _C0;
    uniform vec3 _CameraPos;

    vec3 _Diffuse = vec3({}, {}, {});

    float _BoundingSphereRadius = {}.0;
    float _Threshold = {}.0;
    float _Epsilon = {};

    int _MaxIter = {};
    int _NormalIter = {};

    vec3 intersectSphere(vec3 worldPos,vec3 viewDirection)
    {{
      float B, C, d, t0, t1, t;
      B = 2.0 * dot( worldPos, viewDirection );
      C = dot( worldPos, viewDirection )
        - _BoundingSphereRadius;
      d = sqrt( B * B - 4.0 * C );
      t0 = ( -B + d ) * 0.5;
      t1 = ( -B - d ) * 0.5;
      t = min( t0, t1 );
      worldPos += t * viewDirection;
      return worldPos;
    }}

    vec4 quatMult( vec4 q1, vec4 q2 ) {{
      vec4 r;
      r.x   = q1.x * q2.x - dot( q1.yzw, q2.yzw );
      r.yzw = q1.x * q2.yzw + q2.x * q1.yzw
            + cross( q1.yzw, q2.yzw );
      return r;
    }}

    vec4 quatSq( vec4 q ) {{
      vec4 r;
      r.x   = q.x * q.x - dot( q.yzw, q.yzw );
      r.yzw = 2.0 * q.x * q.yzw;
      return r;
    }}

    void iterateIntersect(
      inout vec4 q,inout vec4 qp,vec4 c
    ) {{
      for( int i = 0; i < _MaxIter; i++ ) {{
        qp = 2.0 * quatMult(q, qp);
        q = quatSq(q) + c;
        if( dot( q, q ) > _Threshold ) {{ break; }}
      }}
    }}

    float intersectQJulia(
      inout vec3 rO, inout vec3 rD, vec4 c
    ) {{
      float dist;
      while( true ) {{
        vec4 z = vec4( rO, 0.0 );
        vec4 zp = vec4( 1.0, 0.0, 0.0, 0.0 );
        iterateIntersect( z, zp, c );
        float normZ = length( z );
        dist = 0.5 * normZ * log( normZ ) / length( zp );
        rO += rD * dist;
        if(    dist < _Epsilon
            || dot(rO, rO) > _BoundingSphereRadius )
          break;
      }}
      return dist;
    }}

    #define DEL 1e-4

    vec3 normEstimate(vec3 p, vec4 c) {{
      vec3 N;
      vec4 qP = vec4( p, 0.0 );
      float gradX, gradY, gradZ;
      vec4 gx1 = qP - vec4( DEL, 0, 0, 0 );
      vec4 gx2 = qP + vec4( DEL, 0, 0, 0 );
      vec4 gy1 = qP - vec4( 0, DEL, 0, 0 );
      vec4 gy2 = qP + vec4( 0, DEL, 0, 0 );
      vec4 gz1 = qP - vec4( 0, 0, DEL, 0 );
      vec4 gz2 = qP + vec4( 0, 0, DEL, 0 );
      for( int i=0; i< _NormalIter; i++ ) {{
        gx1 = quatSq( gx1 ) + c;
        gx2 = quatSq( gx2 ) + c;
        gy1 = quatSq( gy1 ) + c;
        gy2 = quatSq( gy2 ) + c;
        gz1 = quatSq( gz1 ) + c;
        gz2 = quatSq( gz2 ) + c;
      }}
      gradX = length(gx2) - length(gx1);
      gradY = length(gy2) - length(gy1);
      gradZ = length(gz2) - length(gz1);
      N = normalize(vec3( gradX, gradY, gradZ ));
      return N;
    }}

    vec3 Phong( vec3 light, vec3 eye, vec3 pt, vec3 N ) {{
      vec3 diffuse = _Diffuse;
      // shininess of shading
      const float specularExponent = 2.0;
      // amplitude of specular highlight
      const float specularity = 0.05;
      // find the vector to the light
      vec3 L     = normalize( light - pt );
      // find the vector to the eye
      vec3 E     = normalize( eye   - pt );
      // find the cosine of the angle between light and
      // normal
      float NdotL = dot( N, L );
      // find the reflected vector
      vec3 R     = L - 2.0 * NdotL * N;
      diffuse += abs( N ) * 0.3;
      return diffuse * max( NdotL, 0.0 ) + specularity
        * pow( max(dot(E,R), 0.0), specularExponent );
    }}

    void main() {{
      vec3 rD = normalize(world_pos - _CameraPos);
      vec3 rO = intersectSphere(world_pos, rD);

      float dist = intersectQJulia(rO, rD, _C0);

      if (dist < _Epsilon) {{
        vec3 N = normEstimate(rO, _C0);
        //outColor = vec4((N * 0.5 + 0.5), 1.0);
        outColor = vec4(
          Phong(_CameraPos, rD, rO, N),
          1.0
        );
        //outColor = vec4(0.5, 0.0, 0.5, 1.0);
      }} else {{
        outColor = vec4(
          34.0/255.0, 31.0/255.0, 32.0/255.0, 1.0
        );
      }}
    }}
    ",
    random_f32(), random_f32(), random_f32(),
    _BOUNDINGSPHERERADIUS, _THRESHOLD, _EPSILON,
    _MAXITER, _NORMALITER
  )
}
// }}}

// Vector2 {{{
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
struct Vector2 {
  x: f32,
  y: f32,
}

impl Add<Vector2> for Vector2 {
  type Output = Self;

  fn add(self, other: Vector2) -> Self {
    Vector2 { x: self.x + other.x
            , y: self.y + other.y }
  }
}

impl Add<f32> for Vector2 {
  type Output = Self;

  fn add(self, scalar: f32) -> Self {
    Vector2 { x: self.x + scalar
            , y: self.y + scalar }
  }
}

impl Sub<Vector2> for Vector2 {
  type Output = Self;

  fn sub(self, other: Vector2) -> Self {
    Vector2 { x: self.x - other.x
            , y: self.y - other.y }
  }
}

impl Sub<f32> for Vector2 {
  type Output = Self;

  fn sub(self, scalar: f32) -> Self {
    Vector2 { x: self.x - scalar
            , y: self.y - scalar }
  }
}

impl Mul<&Vector2> for &Vector2 {
  type Output = Vector2;

  fn mul(self, other: &Vector2) -> Vector2 {
    Vector2 { x: self.x * other.x
            , y: self.y * other.y }
  }
}

impl Mul<Vector2> for Vector2 {
  type Output = Self;

  fn mul(self, other: Vector2) -> Self {&self * &other}
}

impl Mul<f32> for Vector2 {
  type Output = Vector2;

  fn mul(self, scalar: f32) -> Vector2 {
    Vector2 { x: self.x * scalar
            , y: self.y * scalar }
  }
}

impl Mul<Vector2> for f32 {
  type Output = Vector2;

  fn mul(self, vec: Vector2) -> Vector2 { vec * self }
}

impl From<[f32;2]> for Vector2 {
  fn from(slice: [f32;2]) -> Vector2 {
    Vector2::new(slice[0], slice[1])
  }
}

impl Vector2 {
  fn new(x: f32, y: f32) -> Vector2 {Vector2{ x: x, y: y }}
}
// }}}

// Vector3 {{{
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
struct Vector3 {
  x: f32,
  y: f32,
  z: f32
}

impl Add<Vector3> for Vector3 {
  type Output = Self;

  fn add(self, other: Vector3) -> Self {
    Vector3 { x: self.x + other.x
            , y: self.y + other.y
            , z: self.z + other.z }
  }
}

impl Add<f32> for Vector3 {
  type Output = Self;

  fn add(self, scalar: f32) -> Self {
    Vector3 { x: self.x + scalar
            , y: self.y + scalar
            , z: self.z + scalar }
  }
}

impl Sub<Vector3> for Vector3 {
  type Output = Self;

  fn sub(self, other: Vector3) -> Self {
    Vector3 { x: self.x - other.x
            , y: self.y - other.y
            , z: self.z - other.z }
  }
}

impl Sub<f32> for Vector3 {
  type Output = Self;

  fn sub(self, scalar: f32) -> Self {
    Vector3 { x: self.x - scalar
            , y: self.y - scalar
            , z: self.z - scalar }
  }
}

impl Mul<&Vector3> for &Vector3 {
  type Output = Vector3;

  fn mul(self, other: &Vector3) -> Vector3 {
    Vector3 { x: self.x * other.x
            , y: self.y * other.y
            , z: self.z * other.z }
  }
}

impl Mul<Vector3> for Vector3 {
  type Output = Self;

  fn mul(self, other: Vector3) -> Self {&self * &other}
}

impl Mul<f32> for Vector3 {
  type Output = Vector3;

  fn mul(self, scalar: f32) -> Vector3 {
    Vector3 { x: self.x * scalar
            , y: self.y * scalar
            , z: self.z * scalar }
  }
}

impl From<[f32;3]> for Vector3 {
  fn from(slice: [f32;3]) -> Vector3 {
    Vector3::new(slice[0], slice[1], slice[2])
  }
}

impl Vector3 {
  fn new(x: f32, y: f32, z: f32) -> Vector3 {
    Vector3 { x: x, y: y, z: z }
  }

  fn to_slice(&self) -> [f32;3] {[self.x, self.y, self.z]}
}
// }}}

// Vector4 {{{
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
struct Vector4 {
  x: f32,
  y: f32,
  z: f32,
  w: f32
}

impl Add<Vector4> for Vector4 {
  type Output = Self;

  fn add(self, other: Vector4) -> Self {
    Vector4 { x: self.x + other.x
            , y: self.y + other.y
            , z: self.z + other.z
            , w: self.w + other.w }
  }
}

impl Add<f32> for Vector4 {
  type Output = Self;

  fn add(self, scalar: f32) -> Self {
    Vector4 { x: self.x + scalar
            , y: self.y + scalar
            , z: self.z + scalar
            , w: self.w + scalar }
  }
}

impl Sub<Vector4> for Vector4 {
  type Output = Self;

  fn sub(self, other: Vector4) -> Self {
    Vector4 { x: self.x - other.x
            , y: self.y - other.y
            , z: self.z - other.z
            , w: self.w - other.w }
  }
}

impl Sub<f32> for Vector4 {
  type Output = Self;

  fn sub(self, scalar: f32) -> Self {
    Vector4 { x: self.x - scalar
            , y: self.y - scalar
            , z: self.z - scalar
            , w: self.w - scalar }
  }
}

impl Mul<&Vector4> for &Vector4 {
  type Output = Vector4;

  fn mul(self, other: &Vector4) -> Vector4 {
    Vector4 { x: self.x * other.x
            , y: self.y * other.y
            , z: self.z * other.z
            , w: self.w * other.w }
  }
}

impl Mul<Vector4> for Vector4 {
  type Output = Self;

  fn mul(self, other: Vector4) -> Self {&self * &other}
}

impl Mul<f32> for Vector4 {
  type Output = Vector4;

  fn mul(self, scalar: f32) -> Vector4 {
    Vector4 { x: self.x * scalar
            , y: self.y * scalar
            , z: self.z * scalar
            , w: self.w * scalar }
  }
}

impl From<[f32;4]> for Vector4 {
  fn from(slice: [f32;4]) -> Vector4 {
    Vector4::new(slice[0], slice[1], slice[2], slice[3])
  }
}

impl Vector4 {
  fn new(x: f32, y: f32, z: f32, w: f32) -> Vector4 {
    Vector4 { x: x, y: y, z: z, w: w }
  }

  fn to_slice(&self) -> [f32;4] {
    [self.x, self.y, self.z, self.w]
  }
}
// }}}

// CubicBezier {{{
#[derive(Debug, Serialize, Deserialize)]
struct CubicBezier
  <E: Copy + Add<Output=E> + Mul<f32, Output=E>>
{
  p0: E,
  p1: E,
  p2: E,
  p3: E
}

impl<E: Copy + Add<Output=E> + Mul<f32, Output=E>>
  CubicBezier<E>
{
  fn at(&self, t: f32) -> E {
      self.p0 * (1.0 - t) * (1.0 - t) * (1.0 - t)
    + self.p1 * 3.0       * (1.0 - t) * (1.0 - t) * t
    + self.p2 * 3.0       * (1.0 - t) * t         * t
    + self.p3 * t         * t         * t
  }
}
// }}}

fn random_f32() -> f32 {
  f64::try_from(js!{return Math.random();}).unwrap() as f32
}

fn random_arc() -> f32 { random_f32() * 2.0 * PI }

fn random_vec4_sphere() -> Vector4 {
  let r     = random_f32();
  let alpha = random_arc();
  let beta  = random_arc();
  let gamma = random_arc();
  Vector4::new( r * alpha.cos()
              , r * alpha.sin() * beta.cos()
              , r * alpha.sin() * beta.sin() * gamma.cos()
              , r * alpha.sin() * beta.sin() * gamma.sin())
}

// State {{{

struct ProgramLocations {
  c0:     WebGLUniformLocation,
  camera: WebGLUniformLocation,
  port0:  WebGLUniformLocation,
  port1:  WebGLUniformLocation,
  port2:  WebGLUniformLocation,
  port3:  WebGLUniformLocation,
}


struct Camera {
  pos:   Vector3,
  port0: Vector3,
  port1: Vector3,
  port2: Vector3,
  port3: Vector3,
  anim:  CubicBezier<Vector2>
}

struct Julia {
  current_c0: Vector4,
  next_p1:    Vector4,
  anim:       CubicBezier<Vector4>
}

struct State {
  context:   gl,
  locations: ProgramLocations,
  camera:    Camera,
  t:         f32,
  julia:     Julia,
}

impl State {
  fn render(&mut self, rc: Rc<RefCell<Self>>) {
    if self.t < 1.0 {
      self.julia.current_c0 = self.julia.anim.at(self.t);

      self.context.uniform4fv_1(
        Some(&self.locations.c0),
        &self.julia.current_c0.to_slice() as &[f32],
      );

      let camera_arcs = self.camera.anim.at(self.t);

      let new_camera_pos: Vector3 = Vector3::new(
        CR * camera_arcs.x.sin() * camera_arcs.y.cos(),
        CR * camera_arcs.x.sin() * camera_arcs.y.sin(),
        CR * camera_arcs.x.cos()
      );

      let rotation = q::rotation_from_to(
        self.camera.pos.to_slice(),
        new_camera_pos.to_slice()
      );

      self.camera.pos = new_camera_pos;

      self.camera.port0 = q::rotate_vector(
        rotation,
        self.camera.port0.to_slice()
      ).into();

      self.camera.port1 = q::rotate_vector(
        rotation,
        self.camera.port1.to_slice()
      ).into();

      self.camera.port2 = q::rotate_vector(
        rotation,
        self.camera.port2.to_slice()
      ).into();

      self.camera.port3 = q::rotate_vector(
        rotation,
        self.camera.port3.to_slice()
      ).into();

      self.context.uniform3fv_1(
        Some(&self.locations.camera),
        &self.camera.pos.to_slice() as &[f32],
      );

      self.context.uniform3fv_1(
        Some(&self.locations.port0),
        &self.camera.port0.to_slice() as &[f32],
      );

      self.context.uniform3fv_1(
        Some(&self.locations.port1),
        &self.camera.port1.to_slice() as &[f32],
      );

      self.context.uniform3fv_1(
        Some(&self.locations.port2),
        &self.camera.port2.to_slice() as &[f32],
      );

      self.context.uniform3fv_1(
        Some(&self.locations.port3),
        &self.camera.port3.to_slice() as &[f32],
      );

      let positions = TypedArray::<f32>::from(&[
        self.camera.port0.to_slice(),
        self.camera.port1.to_slice(),
        self.camera.port2.to_slice(),
        self.camera.port3.to_slice(),
      ].concat()[..]).buffer();

      self.context.buffer_data_1( gl::ARRAY_BUFFER
                                , Some(&positions)
                                , gl::STATIC_DRAW );

      self.context.draw_arrays(gl::TRIANGLE_STRIP, 0, 4);

      self.t += 1e-3;

    } else {

      self.t = 0.;

      self.camera.anim = CubicBezier::<Vector2> {
        p0: self.camera.anim.p3,
        p1: 2. * self.camera.anim.p3 - self.camera.anim.p2,
        p2: Vector2::new(random_arc(), random_arc()),
        p3: Vector2::new(random_arc(), random_arc()),
      };

      let next_p1 = random_vec4_sphere();
      let p2      = random_vec4_sphere();

      self.julia.anim = CubicBezier::<Vector4> {
        p0: self.julia.anim.p3,
        p1: self.julia.next_p1,
        p2: p2,
        p3: p2 + (next_p1 - p2) * 0.5,
      };

      self.julia.next_p1 = next_p1;
    }

    window().request_animation_frame(move |_| {
      rc.borrow_mut().render(rc.clone());
    });
  }
}
// }}}

fn main() {
  stdweb::initialize();

  // WebGL2 init {{{

  // context
  let c: CanvasElement = document().query_selector("#c")
    .unwrap().unwrap().try_into().unwrap();

  c.set_width(920 as u32);
  c.set_height(920 as u32);

  let context: gl = match c.get_context() {
    Ok(context) => context,
    Err(_)      => {
      js! {
        console.log("no webgl2 support!");
        fallback();
      }
      panic!();
    }
  };

  // init shaders
  let vertex_shader = context
    .create_shader(gl::VERTEX_SHADER).unwrap();
  context.shader_source(&vertex_shader, VERT_SRC);
  context.compile_shader(&vertex_shader);

  let fragment_shader = context
    .create_shader(gl::FRAGMENT_SHADER).unwrap();

  context.shader_source(&fragment_shader, &frag_src());
  context.compile_shader(&fragment_shader);

  // link shader to webgl2 program
  let program = context.create_program().unwrap();
  context.attach_shader(&program, &vertex_shader);
  context.attach_shader(&program, &fragment_shader);
  context.link_program(&program);

  // create and bind vertex buffer
  let vertex_buffer = context.create_buffer().expect("e");
  context.bind_buffer( gl::ARRAY_BUFFER
                     , Some(&vertex_buffer));

  let vertex_array_object = context.create_vertex_array()
    .unwrap();
  context.bind_vertex_array(Some(&vertex_array_object));

  // vertices for the vertex shader
  let vertex_location = context
    .get_attrib_location(&program, "a_position") as u32;
  context.enable_vertex_attrib_array(vertex_location);

  let size: i32   = 3;
  let type_: u32  = gl::FLOAT;
  let normalize   = false;
  let stride: i32 = 0;
  let offset: i64 = 0;

  context.vertex_attrib_pointer( vertex_location, size
                               , type_, normalize, stride
                               , offset );

  context.clear_color(0.0, 0.0, 0.0, 1.0);
  context.clear(
    gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT
  );

  context.blend_func(gl::ONE, gl::ONE_MINUS_SRC_ALPHA);

  context.use_program(Some(&program));
  // }}}

  let c0 = context
    .get_uniform_location(&program, "_C0").unwrap();
  let camera = context
    .get_uniform_location(&program, "_CameraPos").unwrap();

  let port0 = context
    .get_uniform_location(&program, "port0").unwrap();
  let port1 = context
    .get_uniform_location(&program, "port1").unwrap();
  let port2 = context
    .get_uniform_location(&program, "port2").unwrap();
  let port3 = context
    .get_uniform_location(&program, "port3").unwrap();

  // animatation

  let c0_v = Vector4::new(0., 0., 0., 0.);

  context.uniform4fv_1( Some(&c0)
                      , &c0_v.to_slice() as &[f32] );

  let camera_anim = CubicBezier::<Vector2> {
    p0: Vector2::new(0., 0.),
    p1: Vector2::new(random_arc(), random_arc()),
    p2: Vector2::new(random_arc(), random_arc()),
    p3: Vector2::new(random_arc(), random_arc()),
  };

  let next_p1 = random_vec4_sphere();
  let p2      = random_vec4_sphere();

  let julia_anim = CubicBezier::<Vector4> {
    p0: c0_v,
    p1: random_vec4_sphere(),
    p2: p2,
    p3: p2 + (next_p1 - p2) * 0.5,
  };

  let locations = ProgramLocations { c0:     c0
                                   , camera: camera
                                   , port0:  port0
                                   , port1:  port1
                                   , port2:  port2
                                   , port3:  port3 };

  let camera = Camera { pos:   Vector3::new( 0.,  0., CR)
                      , port0: Vector3::new(-1., -1., 0.)
                      , port1: Vector3::new(-1.,  1., 0.)
                      , port2: Vector3::new( 1., -1., 0.)
                      , port3: Vector3::new( 1.,  1., 0.)
                      , anim:  camera_anim };

  let julia = Julia { current_c0: c0_v
                    , next_p1:    next_p1
                    , anim:       julia_anim };

  let state = Rc::new(RefCell::new(State {
    context:   context,
    locations: locations,
    camera:    camera,
    t:         0.,
    julia:     julia,
  }));

  state.borrow_mut().render(state.clone());

  stdweb::event_loop();
}
