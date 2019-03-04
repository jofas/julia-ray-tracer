using static System.Math;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Controller : MonoBehaviour
{
  // boundries for the prison point
  public float r_min = 0.2f, r_max = 2f;
  public float julia_transform = 0.01f;

  public float camera_transform = 0.1f;
  private const float camera_radius = 3.3f;

  // camera zoom
  public float zoom_transform = 0.1f;

  [Range(1,100)]
  public int camera_zoom_change = 40;
  private int camera_zoom_change_steps = 0;

  private bool zoom_steps_dir;

  private const float field_of_view_min = 40f;
  private const float field_of_view_max = 60f;

  // compute new camera_dir after {camera_dir_change} many
  // steps
  [Range(1,1000)]
  public int camera_dir_change = 400;
  private int camera_dir_change_steps = 0;

  public int frame_rate = 20;

  // prison point for the julia set
  private Vector4 c0 = new Vector4();

  // vector we add to c0 for transition
  private Vector4 julia_dir  = new Vector4();

  // vector we add to the camera position
  // (!) we always want to stay on the same cube (r)
  //     -> v = ( r / |v| ) * v
  private Vector3 camera_dir = new Vector3();

  private float pi = (float) PI;

  private Material material;
  private Camera camera;
  private Transform camera_trans;

  // Start is called before the first frame update
  void Start()
  {
    QualitySettings.vSyncCount = 0;
    Application.targetFrameRate = this.frame_rate;

    set_material();

    var co = GameObject.Find("Main Camera");
    camera = co.GetComponent<Camera>();
    camera_trans = camera.GetComponent<Transform>();

    // init camera zoom
    zoom_steps_dir = Random.Range(0f, 1f) > 0.5f;

    camera.fieldOfView = Random.Range( field_of_view_min
                                     , field_of_view_max );

    // init camera position
    camera_dir =
      random_point_on_3d_sphere(camera_transform);

    camera_trans.position =
      random_point_on_3d_sphere(camera_radius);

    camera_trans.rotation = Quaternion.LookRotation(
      - camera_trans.position
    );

    // init julia set
    julia_dir = random_point_on_4d_sphere(julia_transform);
    c0 = random_prison_point();

    // set the julia set
    material.SetVector("_Mu", c0);
  }

  // Update is called once per frame
  void Update()
  {
    if (Input.GetKey("escape")){Application.Quit();}

    // camera zoom
    if (zoom_steps_dir) {
      change_field_of_view(
        camera.fieldOfView + zoom_transform,
        camera.fieldOfView + zoom_transform
        < field_of_view_max
      );
    } else {
      change_field_of_view(
        camera.fieldOfView - zoom_transform,
        camera.fieldOfView - zoom_transform
        > field_of_view_min
      );
    }

    // camera
    var np = camera_trans.position + camera_dir;
    np = (camera_radius / np.magnitude) * np;

    camera_trans.position = np;

    camera_trans.rotation = Quaternion.LookRotation(
      - camera_trans.position
    );

    camera_dir_change_steps += 1;

    if (camera_dir_change_steps == camera_dir_change) {
      camera_dir =
        random_point_on_3d_sphere(camera_transform);
      camera_dir_change_steps = 0;
    }

    // julia
    c0 += julia_dir;
    while( c0.magnitude > this.r_max
        || c0.magnitude < this.r_min )
    {
      julia_dir =
        random_point_on_4d_sphere(julia_transform);
      c0 += julia_dir;
    }
    material.SetVector("_Mu", c0);
  }

  void set_material() {
    var mesh_r = GetComponent<MeshRenderer>();
    if (mesh_r == null)
      mesh_r = gameObject.AddComponent<MeshRenderer> ();

    if (mesh_r.sharedMaterial == null) {
      mesh_r.sharedMaterial = material =
        new Material(Shader.Find("___Shader"));

    }
    material = mesh_r.sharedMaterial;
  }

  void change_field_of_view( float new_field_of_view
                           , bool in_boundries       )
  {
    if ( !in_boundries
      || camera_zoom_change_steps == camera_zoom_change )
    {
      zoom_steps_dir = !zoom_steps_dir;
      camera_zoom_change_steps = 0;
    } else {
      camera.fieldOfView = new_field_of_view;
    }
  }

  Vector4 random_prison_point() {
    var r = ( this.r_max - this.r_min )
          * Random.Range(0f, 1f)
          + this.r_min;
    return random_point_on_4d_sphere(r);
  }

  Vector3 random_point_on_3d_sphere(float r) {
    var p = new Vector3();

    var alpha = Random.Range(0f, 2f * this.pi);
    var beta  = Random.Range(0f, 2f * this.pi);

    var c_a = (float) Cos(alpha);
    var c_b = (float) Cos(beta);

    var s_a = (float) Sin(alpha);
    var s_b = (float) Sin(beta);

    p.x = r * c_a;
    p.y = r * s_a * c_b;
    p.z = r * s_a * s_b;

    return p;
  }

  Vector4 random_point_on_4d_sphere(float r) {
    var p = new Vector4();

    var alpha = Random.Range(0f, 2f * this.pi);
    var beta  = Random.Range(0f, 2f * this.pi);
    var gamma = Random.Range(0f, 2f * this.pi);

    var c_a = (float) Cos(alpha);
    var c_b = (float) Cos(beta);
    var c_g = (float) Cos(gamma);

    var s_a = (float) Sin(alpha);
    var s_b = (float) Sin(beta);
    var s_g = (float) Sin(gamma);

    p.x = r * c_a;
    p.y = r * s_a * c_b;
    p.z = r * s_a * s_b * c_g;
    p.w = r * s_a * s_b * s_g;

    return p;
  }

  float deg_to_rad(int deg) {
    return ( (float) deg ) * this.pi / 180f;
  }
}
