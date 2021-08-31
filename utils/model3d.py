"""
Written by Haoshu Fang, 
modified from ssd-6d(https://github.com/wadimkehl/ssd-6d) by Wadim Kehl
"""

import os
import OpenGL.GL as gl
import numpy as np
from scipy.spatial.distance import pdist
from plyfile import PlyData
from cv2 import cv2
from vispy import app, gloo
import dill

app.use_app('pyqt5')


def loadmodel(modeldir,modelname):
    if os.path.exists(os.path.join(modeldir,modelname+'.cache')):
        with open(os.path.join(modeldir,modelname+'.cache'),'rb') as f:
            model = dill.load(f)
    else:
        model = Model3D(os.path.join(modeldir,modelname))
    return model    

def cachemodel(modeldir,modelname,model):
    assert os.path.exists(modeldir)
    with open(os.path.join(modeldir,modelname+'.cache'),'wb') as f:
        dill.dump(model,f)


class Model3D:
    def __init__(self, file_to_load=None):
        self.vertices = None
        self.centroid = None
        self.indices = None
        self.colors = None
        self.texcoord = None
        self.texture = None
        self.collated = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.bb = None
        self.bb_vbuffer = None
        self.bb_ibuffer = None
        self.diameter = None
        if file_to_load:
            self.load(file_to_load)

    def _compute_bbox(self):

        self.bb = []
        minx, maxx = min(self.vertices[:, 0]), max(self.vertices[:, 0])
        miny, maxy = min(self.vertices[:, 1]), max(self.vertices[:, 1])
        minz, maxz = min(self.vertices[:, 2]), max(self.vertices[:, 2])
        self.bb.append([minx, miny, minz])
        self.bb.append([minx, maxy, minz])
        self.bb.append([minx, miny, maxz])
        self.bb.append([minx, maxy, maxz])
        self.bb.append([maxx, miny, minz])
        self.bb.append([maxx, maxy, minz])
        self.bb.append([maxx, miny, maxz])
        self.bb.append([maxx, maxy, maxz])
        self.bb = np.asarray(self.bb, dtype=np.float32)
        self.diameter = max(pdist(self.bb, 'euclidean'))

        # Set up rendering data
        colors = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1],
                  [0, 1, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        indices = [
            0, 1, 0, 2, 3, 1, 3, 2, 4, 5, 4, 6, 7, 5, 7, 6, 0, 4, 1, 5, 2, 6,
            3, 7
        ]

        vertices_type = [('a_position', np.float32, 3),
                         ('a_color', np.float32, 3)]
        collated = np.asarray(list(zip(self.bb, colors)), vertices_type)
        self.bb_vbuffer = gloo.VertexBuffer(collated)
        self.bb_ibuffer = gloo.IndexBuffer(indices)

    def load(self, path, demean=False, scale=1.0):
        data = PlyData.read(path)
        self.vertices = np.zeros((data['vertex'].count, 3))
        self.vertices[:, 0] = np.array(data['vertex']['x'])
        self.vertices[:, 1] = np.array(data['vertex']['y'])
        self.vertices[:, 2] = np.array(data['vertex']['z'])
        self.vertices *= scale
        self.centroid = np.mean(self.vertices, 0)

        if demean:
            self.centroid = np.zeros((1, 3), np.float32)
            self.vertices -= self.centroid

        self._compute_bbox()

        self.indices = np.asarray(
            list(data['face']['vertex_indices']), np.uint32)

        # Look for texture map as jpg or png
        filename = path.split('/')[-1]
        abs_path = path[:path.find(filename)]
        tex_to_load = None
        if os.path.exists(abs_path + filename[:-4] + '.jpg'):
            tex_to_load = abs_path + filename[:-4] + '.jpg'
        elif os.path.exists(abs_path + filename[:-4] + '.png'):
            tex_to_load = abs_path + filename[:-4] + '.png'

        # Try to read out texture coordinates
        if tex_to_load is not None:
            print('Loading {} with texture {}'.format(filename, tex_to_load))
            image = cv2.flip(cv2.imread(tex_to_load, cv2.IMREAD_UNCHANGED),
                             0)  # Must be flipped because of OpenGL
            self.texture = gloo.Texture2D(image)

            # If texcoords are face-wise
            if 'texcoord' in str(data):
                self.texcoord = np.asarray(list(data['face']['texcoord']))
                assert self.indices.shape[0] == self.texcoord.shape[
                    0]  # Check same face count
                temp = np.zeros((data['vertex'].count, 2))
                temp[self.indices.flatten()] = self.texcoord.reshape((-1, 2))
                self.texcoord = temp

            # If texcoords are vertex-wise
            elif 'texture_u' in str(data):
                self.texcoord = np.zeros((data['vertex'].count, 2))
                self.texcoord[:, 0] = np.array(data['vertex']['texture_u'])
                self.texcoord[:, 1] = np.array(data['vertex']['texture_v'])

        # If texture coords loaded succesfully
        if self.texcoord is not None:
            vertices_type = [('a_position', np.float32, 3),
                             ('a_texcoord', np.float32, 2)]
            self.collated = np.asarray(
                list(zip(self.vertices, self.texcoord)), vertices_type)

        # Otherwise fall back to vertex colors
        else:
            self.colors = 0.5 * np.ones((data['vertex'].count, 3))
            if 'blue' in str(data):
                print('Loading {} with vertex colors'.format(filename))
                self.colors[:, 0] = np.array(data['vertex']['blue'])
                self.colors[:, 1] = np.array(data['vertex']['green'])
                self.colors[:, 2] = np.array(data['vertex']['red'])
                self.colors /= 255.0
            else:
                print('Loading {} without any coloring!!'.format(filename))
            vertices_type = [('a_position', np.float32, 3),
                             ('a_color', np.float32, 3)]
            self.collated = np.asarray(
                list(zip(self.vertices, self.colors)), vertices_type)

        self.vertex_buffer = gloo.VertexBuffer(self.collated)
        self.index_buffer = gloo.IndexBuffer(self.indices.flatten())

_vertex_code_colored = """
uniform mat4 u_mv; 
uniform mat4 u_mvp; 
uniform vec3 u_light_eye_pos; 
 
attribute vec3 a_position; 
attribute vec3 a_color; 
 
varying vec3 v_color; 
varying vec3 v_eye_pos; 
varying vec3 v_L; 
 
void main() { 
    gl_Position = u_mvp * vec4(a_position, 1.0); 
    v_color = a_color; 
    v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex position in eye coordinates 
    v_L = normalize(u_light_eye_pos - v_eye_pos); // Vector to the light 
} 
"""

_fragment_code_colored = """
uniform float u_light_ambient_w; 
varying vec3 v_color; 
varying vec3 v_eye_pos; 
varying vec3 v_L; 
 
void main() { 
    // Face normal in eye coordinates 
    vec3 face_normal = normalize(cross(dFdx(v_eye_pos), dFdy(v_eye_pos))); 
 
    float light_diffuse_w = max(dot(normalize(v_L), normalize(face_normal)), 0.0); 
    float light_w = u_light_ambient_w + 0.5 * light_diffuse_w;
    if(light_w > 1.0) light_w = 1.0; 
    gl_FragColor = vec4(light_w * v_color, 1.0); 
} 
"""

_vertex_code_textured = """
uniform mat4 u_mv; 
uniform mat4 u_mvp; 
uniform vec3 u_light_eye_pos; 
attribute vec3 a_position; 
attribute vec2 a_texcoord; 
varying vec2 v_texcoord; 
varying vec3 v_eye_pos; 
varying vec3 v_L; 
void main() { 
    gl_Position = u_mvp * vec4(a_position, 1.0); 
    v_texcoord = a_texcoord;
    v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex position in eye coordinates 
    v_L = normalize(u_light_eye_pos - v_eye_pos); // Vector to the light 
} 
"""

_fragment_code_textured = """
uniform float u_light_ambient_w; 
uniform sampler2D u_tex;
varying vec2 v_texcoord; 
varying vec3 v_eye_pos; 
varying vec3 v_L; 
void main() { 
    // Face normal in eye coordinates 
    vec3 face_normal = normalize(cross(dFdx(v_eye_pos), dFdy(v_eye_pos))); 
    float light_diffuse_w = max(dot(normalize(v_L), normalize(face_normal)), 0.0); 
    float light_w = u_light_ambient_w + 0.5 * light_diffuse_w;
    if(light_w > 1.0) light_w = 1.0; 
    gl_FragColor = texture2D(u_tex, v_texcoord) * light_w;
} 
"""

def draw_model(image, pose, cam, model):
    """render objects onto resized image
    Args: 
        image: Numpy array, hxwx3 uint8
        pose: 4x4 transformation matrix
        cam: Intrinsics for rendering
        model: Model3D object that has field diameter
    Returns: 
        Rendered image
    """
    image = image.astype(np.float) / 255.0
    ren = Renderer((image.shape[1], image.shape[0]), cam)
    ren.clear()
    ren.set_cam(cam)
    out = np.copy(image)

    ren.draw_model(model, pose)
    col, dep = ren.finish()
    # Copy the rendering over into the scene
    mask = np.dstack((dep, dep, dep)) > 0
    out[mask] = col[mask]
    out = (out * 255.0).astype(np.uint8)
    del ren
    return out

def singleton(cls):
    instances = {}

    def get_instance(size, cam):
        if cls not in instances:
            instances[cls] = cls(size, cam)
        return instances[cls]

    return get_instance

@singleton  # Don't throw GL context into trash when having more than one Renderer instance
class Renderer(app.Canvas):
    def __init__(self, size, cam):
        config = dict(red_size=8,
                      green_size=8,
                      blue_size=8,
                      alpha_size=8,
                      depth_size=24,
                      stencil_size=0,
                      double_buffer=True,
                      stereo=False,
                      samples=0)

        app.Canvas.__init__(self, show=False, size=size, config=config)
        self.shape = (size[1], size[0])
        self.yz_flip = np.eye(4, dtype=np.float32)
        self.yz_flip[1, 1], self.yz_flip[2, 2] = -1, -1

        self.set_cam(cam)
        # Set up shader programs
        self.program_col = gloo.Program(_vertex_code_colored,
                                        _fragment_code_colored)
        self.program_tex = gloo.Program(_vertex_code_textured,
                                        _fragment_code_textured)

        # Texture where we render the color/depth and its FBO
        self.col_tex = gloo.Texture2D(shape=self.shape + (3, ))
        self.fbo = gloo.FrameBuffer(self.col_tex,
                                    gloo.RenderBuffer(self.shape))
        self.fbo.activate()
        gloo.set_state(depth_test=True, blend=False, cull_face=True)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gloo.set_clear_color((0.0, 0.0, 0.0))
        gloo.set_viewport(0, 0, *self.size)

    def set_cam(self, cam, clip_near=0.01, clip_far=10.0):
        self.cam = cam
        self.clip_near = clip_near
        self.clip_far = clip_far
        self.mat_proj = self.build_projection(cam, 0, 0, self.shape[1],
                                              self.shape[0], clip_near,
                                              clip_far)

    def clear(self):
        gloo.clear(color=True, depth=True)

    def finish(self):

        im = gl.glReadPixels(0, 0, self.size[0], self.size[1], gl.GL_RGB,
                             gl.GL_FLOAT)
        rgb = np.copy(np.frombuffer(
            im, np.float32)).reshape(self.shape +
                                     (3, ))[::-1, :]  # Read buffer and flip Y
        im = gl.glReadPixels(0, 0, self.size[0], self.size[1],
                             gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        dep = np.copy(np.frombuffer(
            im, np.float32)).reshape(self.shape +
                                     (1, ))[::-1, :]  # Read buffer and flip Y

        # Convert z-buffer to depth map
        mult = (self.clip_near * self.clip_far) / (self.clip_near -
                                                   self.clip_far)
        addi = self.clip_far / (self.clip_near - self.clip_far)
        bg = dep == 1
        dep = mult / (dep + addi)
        dep[bg] = 0
        return rgb, np.squeeze(dep)

    def draw_model(self, model, pose, ambient_weight=0.5, light=(0, 0, 0)):

        # View matrix (transforming the coordinate system from OpenCV to OpenGL camera space)
        mv = (self.yz_flip.dot(pose)
              ).T  # OpenCV to OpenGL camera system (flipped, column-wise)
        mvp = mv.dot(self.mat_proj)

        used_program = self.program_col
        if model.texcoord is not None:
            used_program = self.program_tex
            used_program['u_tex'] = model.texture

        used_program.bind(model.vertex_buffer)
        used_program['u_light_eye_pos'] = light
        used_program['u_light_ambient_w'] = ambient_weight
        used_program['u_mv'] = mv
        used_program['u_mvp'] = mvp
        used_program.draw('triangles', model.index_buffer)

    def draw_boundingbox(self, model, pose, color=None):

        # View matrix (transforming the coordinate system from OpenCV to OpenGL camera space)
        mv = (self.yz_flip.dot(pose)
              ).T  # OpenCV to OpenGL camera system (flipped, column-wise)
        mvp = mv.dot(self.mat_proj)

        self.program_col.bind(model.bb_vbuffer)
        self.program_col['u_light_eye_pos'] = (0, 0, 0)
        self.program_col['u_light_ambient_w'] = 1
        self.program_col['u_mv'] = mv
        self.program_col['u_mvp'] = mvp
        self.program_col['a_color'] = color
        self.program_col.draw('lines', model.bb_ibuffer)

    def build_projection(self, cam, x0, y0, w, h, nc, fc):

        q = -(fc + nc) / float(fc - nc)
        qn = -2 * (fc * nc) / float(fc - nc)

        # Draw our images upside down, so that all the pixel-based coordinate systems are the same
        proj = np.array([
            [
                2 * cam[0, 0] / w, -2 * cam[0, 1] / w,
                (-2 * cam[0, 2] + w + 2 * x0) / w, 0
            ],
            [0, -2 * cam[1, 1] / h, (-2 * cam[1, 2] + h + 2 * y0) / h, 0],
            [
                0, 0, q, qn
            ],  # This row is standard glPerspective and sets near and far planes
            [0, 0, -1, 0]
        ])

        # Compensate for the flipped image
        proj[1, :] *= -1.0
        return proj.T

    def compute_metrical_clip(self, pose, diameter):

        width = self.cam[0, 0] * diameter / pose[2,
                                                 3]  # X coordinate == shape[1]
        height = self.cam[1, 1] * diameter / pose[
            2, 3]  # Y coordinate == shape[0]
        proj = np.matmul(self.cam, pose[0:3, 3])
        proj /= proj[2]
        cut = np.asarray([
            proj[1] - height // 2, proj[0] - width // 2, proj[1] + height // 2,
            proj[0] + width // 2
        ],
                         dtype=int)

        # Can lead to offsetted extractions, not really nice...
        cut[0] = np.clip(cut[0], 0, self.shape[0])
        cut[2] = np.clip(cut[2], 0, self.shape[0])
        cut[1] = np.clip(cut[1], 0, self.shape[1])
        cut[3] = np.clip(cut[3], 0, self.shape[1])
        return cut

    def render_view_metrical_clip(self, model, pose, diameter):

        cut = self.compute_metrical_clip(pose, diameter)
        self.clear()
        self.draw_model(model, pose)
        col, dep = self.finish()
        return col[cut[0]:cut[2], cut[1]:cut[3]], dep[cut[0]:cut[2], cut[1]:
                                                      cut[3]]