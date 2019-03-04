// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "___Shader"
{
    Properties
    {
        _ColorCube ("Color cube", Color) = (1, 1, 1, 1)

        _BoundingSphereRadius ("Bounding Sphere Radius", float) = 3.0
        _Threshold("Escape Threshold", float) = 1.0
        _Mu ("Prison Element", vector) = (1.0, 0.0, 0.0, 0.0)
        _Epsilon ("Epsilon", float) = 0.1
        _MaxIter ("Max Iter", int) = 12
        _NormalIter ("Normal Iter", int) = 12
    }
    SubShader
    {
        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            // Ported Keenan Crane's implementation for Unity:
            // https://www.cs.cmu.edu/~kmcrane/Projects/QuaternionJulia/paper.pdf

            struct v2f {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD1;
            };

            v2f vert(appdata_base v) {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                return o;
            }

            fixed4 _ColorCube;

            float _BoundingSphereRadius;
            float _Threshold;
            fixed4 _Mu;
            float _Epsilon;
            int _MaxIter;
            int _NormalIter;

            float4 quatMult( float4 q1, float4 q2 ) {
              float4 r;
              r.x   = q1.x*q2.x - dot( q1.yzw, q2.yzw );
              r.yzw = q1.x*q2.yzw + q2.x*q1.yzw + cross( q1.yzw, q2.yzw );
              return r;
            }

            float4 quatSq( float4 q ) {
              float4 r;
              r.x   = q.x*q.x - dot( q.yzw, q.yzw );
              r.yzw = 2*q.x*q.yzw;
              return r;
            }

            void iterateIntersect( inout float4 q, inout float4 qp, float4 c ) {
              for( int i=0; i<_MaxIter; i++ ) {
                qp = 2.0 * quatMult(q, qp);
                q = quatSq(q) + c;
                if( dot( q, q ) > _Threshold ) { break; }
              }
            }

            float intersectQJulia( inout float3 rO, inout float3 rD, float4 c ) {
              float dist;
              while( 1 ) {
                float4 z = float4( rO, 0 );
                float4 zp = float4( 1, 0, 0, 0 );
                iterateIntersect( z, zp, c );
                float normZ = length( z );
                dist = 0.5 * normZ * log( normZ ) / length( zp );
                rO += rD * dist;
                if( dist < _Epsilon || dot(rO, rO) > _BoundingSphereRadius )
                  break;
              }
              return dist;
            }

            float3 intersectSphere( float3 worldPos, float3 viewDirection ) {
              float B, C, d, t0, t1, t;
              B = 2 * dot( worldPos, viewDirection );
              C = dot( worldPos, viewDirection ) - _BoundingSphereRadius;
              d = sqrt( B*B - 4*C );
              t0 = ( -B + d ) * 0.5;
              t1 = ( -B - d ) * 0.5;
              t = min( t0, t1 );
              worldPos += t * viewDirection;
              return worldPos;
            }

            #define DEL 1e-4

            float3 normEstimate(float3 p, float4 c) {
              float3 N;
              float4 qP = float4( p, 0 );
              float gradX, gradY, gradZ;
              float4 gx1 = qP - float4( DEL, 0, 0, 0 );
              float4 gx2 = qP + float4( DEL, 0, 0, 0 );
              float4 gy1 = qP - float4( 0, DEL, 0, 0 );
              float4 gy2 = qP + float4( 0, DEL, 0, 0 );
              float4 gz1 = qP - float4( 0, 0, DEL, 0 );
              float4 gz2 = qP + float4( 0, 0, DEL, 0 );
              for( int i=0; i< _NormalIter; i++ ) {
                gx1 = quatSq( gx1 ) + c;
                gx2 = quatSq( gx2 ) + c;
                gy1 = quatSq( gy1 ) + c;
                gy2 = quatSq( gy2 ) + c;
                gz1 = quatSq( gz1 ) + c;
                gz2 = quatSq( gz2 ) + c;
              }
              gradX = length(gx2) - length(gx1);
              gradY = length(gy2) - length(gy1);
              gradZ = length(gz2) - length(gz1);
              N = normalize(float3( gradX, gradY, gradZ ));
              return N;
            }

            float3 Phong( float3 light, float3 eye, float3 pt, float3 N ) {
              float3 diffuse = float3( 0.24, 0.36, 0.49 ); // base color of shading
              const int specularExponent = 2;             // shininess of shading
              const float specularity = 0.45;              // amplitude of specular highlight
              float3 L     = normalize( light - pt );  // find the vector to the light
              float3 E     = normalize( eye   - pt );  // find the vector to the eye
              float  NdotL = dot( N, L );              // find the cosine of the angle between light and normal
              float3 R     = L - 2 * NdotL * N;        // find the reflected vector
              diffuse += abs( N )*0.3;  // add some of the normal to the
              return diffuse * max( NdotL, 0 ) + specularity*pow( max(dot(E,R),0), specularExponent );
            }

            fixed4 frag(v2f i) : SV_Target {
              float3 viewDirection = normalize(
                i.worldPos - _WorldSpaceCameraPos
              );

              float3 worldPos = intersectSphere(i.worldPos, viewDirection);

              float dist = intersectQJulia( worldPos, viewDirection, _Mu );

              float4 color = _ColorCube;

              if (dist < _Epsilon) {
                float3 N = normEstimate( worldPos, _Mu );
                // use the normals as color
                //color = float4((N * 0.5 + 0.5), 1.0);
                color = float4(Phong( float3(4.0, 4.0, 2.0), viewDirection, worldPos, N), 1.0);
              }
              return color;
            }

            ENDCG
        }
    }
}
