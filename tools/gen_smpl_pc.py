import os, sys
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import imageio
import struct
from tqdm import tqdm

## These functions are directly borrowed from PIFu implementation
def load_obj_mesh(mesh_file, with_normal=False, with_texture=False, with_texture_image=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
        elif 'mtllib' in line.split():
            mtlname = line.split()[-1]
            mtlfile = os.path.join(os.path.dirname(mesh_file), mtlname)
            with open(mtlfile, 'r') as fmtl:
                mtllines = fmtl.readlines()
                for mtlline in mtllines:
                    # if mtlline.startswith('map_Kd'):
                    if 'map_Kd' in mtlline.split():
                        texname = mtlline.split()[-1]
                        texfile = os.path.join(os.path.dirname(mesh_file), texname)
                        texture_image = cv2.imread(texfile)
                        texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
                        break

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        if with_texture_image:
            return vertices, faces, norms, face_normals, uvs, face_uvs, texture_image
        else:
            return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        # norms = np.array(norm_data)
        # norms = normalize_v3(norms)
        # face_normals = np.array(face_norm_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm

def index(feat, uv, mode='bilinear'):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    # if torch.__version__ >= "1.3.0":
    #     samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    # else:
    samples = torch.nn.functional.grid_sample(feat, uv, mode=mode)
    return samples[:, :, :, 0]  # [B, C, N]


def perspective(points, w2c, camera):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4/9] Tensor of projection matrix
    :param transforms: [Bx4x4] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = w2c[:, :3, :3]
    trans = w2c[:, :3, 3:4]
    points = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy  = points[:,:2, :] / torch.clamp(points[:,2:3,:], 1e-9)
    xy = camera[:,0:2,None]*xy + camera[:,2:4,None]
    points[:,:2, :] = xy
    return points

def load_ply(file_name):
    try:
        fid = open(file_name, 'r')
        head = fid.readline().strip()
        readl = lambda f: f.readline().strip()
    except UnicodeDecodeError as e:
        fid = open(file_name, 'rb')
        readl =	(lambda f: str(f.readline().strip())[2:-1]) \
            if sys.version_info[0] == 3 else \
            (lambda f: str(f.readline().strip()))
        head = readl(fid)
    form = readl(fid).split(' ')[1]
    line = readl(fid)
    vshape = fshape = [0]
    while line != 'end_header':
        s = [i for i in line.split(' ') if len(i) > 0]
        if len(s) > 2 and s[0] == 'element' and s[1] == 'vertex':
            vshape = [int(s[2])]
            line = readl(fid)
            s = [i for i in line.split(' ') if len(i) > 0]
            while s[0] == 'property' or s[0][0] == '#':
                if s[0][0] != '#':
                    vshape += [s[1]]
                line = readl(fid)
                s = [i for i in line.split(' ') if len(i) > 0]
        elif len(s) > 2 and s[0] == 'element' and s[1] == 'face':
            fshape = [int(s[2])]
            line = readl(fid)
            s = [i for i in line.split(' ') if len(i) > 0]
            while s[0] == 'property' or s[0][0] == '#':
                if s[0][0] != '#':
                    fshape = [fshape[0],s[2],s[3]]
                line = readl(fid)
                s = [i for i in line.split(' ') if len(i) > 0]
        else:
            line = readl(fid)
    if form.lower() == 'ascii':
        v = []
        for i in range(vshape[0]):
            s = [i for i in readl(fid).split(' ') if len(i) > 0]
            if s[0][0] != '#':
                v += [[float(i) for i in s]]
        v = np.array(v, np.float32)
        tri = []
        for i in range(fshape[0]):
            s = [i for i in readl(fid).split(' ') if len(i) > 0]
            if s[0][0] != '#':
                tri += [[int(s[1]),int(s[i-1]),int(s[i])] \
                    for i in range(3,len(i))]
        tri = np.array(tri, np.int64)
    else:
        maps = {'float':('f',4), 'double':('d',8), \
            'uint': ('I',4), 'int':   ('i',4), \
            'ushot':('H',2), 'short': ('h',2), \
            'uchar':('B',1), 'char':  ('b',1)}
        if 'little' in form.lower():
            fmt = '<' + ''.join([maps[i][0] for i in vshape[1:]]*vshape[0])
        else:
            fmt = '>' + ''.join([maps[i][0] for i in vshape[1:]]*vshape[0])
        l = sum([maps[i][1] for i in vshape[1:]]) * vshape[0]
        v = struct.unpack(fmt, fid.read(l))
        v = np.array(v).reshape(vshape[0],-1).astype(np.float32)
        tri = []
        for i in range(fshape[0]):
            l = struct.unpack(fmt[0]+maps[fshape[1]][0], fid.read(maps[fshape[1]][1]))
            l = l[0]
            f = struct.unpack(fmt[0]+maps[fshape[2]][0]*l, \
                fid.read(l*maps[fshape[2]][1]))
            tri += [[f[0],f[i-1],f[i]] for i in range(2,len(f))]
        tri = np.array(tri).reshape(fshape[0],-1).astype(np.int64)
    fid.close()
    return v, tri

def save_obj_mesh_with_color(mesh_path, verts, colors, faces=None):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    if faces is not None:
        for f in faces:
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def lumigraph_blending(verts, normals, colors, calibs, s=1):
    """
    Lumigraph Blending (http://cs.harvard.edu/~sjg/papers/ulr.pdf)
    
    verts: points for coloring
    normals: vertices' normal
    colors: per-view color observation
    calibs: calibration matrix
    s: constant coefficient control screw of blending weight / view inner product
    """
    center = verts.mean(0)
    view_dirs = normals
    # view_dirs[:, 2] = 0         # look horizontally to the z axis
    view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-10) # [N, 3]
    attdirs = -torch.inverse(calibs)[:, :3, 2]
    attdirs = attdirs[None,...].repeat(view_dirs.shape[0], 1, 1)
    view_dirs = view_dirs[:,None,:].repeat(1, calibs.shape[0], 1)
    weights = torch.exp(s * (torch.sum(view_dirs*attdirs, dim=-1) - 1))

    w_min = torch.min(weights, dim=-1, keepdim=True)[0]
    weights = weights - w_min
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)  # [N_rand*N_sample, 4]

    colors = colors.permute([2,0,1])
    colors = torch.sum(colors * weights[..., None], dim=1)
    return colors


def pointcloud_upsampling(verts, faces, normals, ratio=5):
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces).to(verts.device)
    coeff = torch.rand((faces.shape[0], ratio, 3)).to(verts.device).float()
    coeff = coeff / (torch.sum(coeff, dim=-1, keepdim=True) + 1e-8)
    idx1 = faces[:, 0].long()
    idx2 = faces[:, 1].long()
    idx3 = faces[:, 2].long()
    coeff1 = coeff[:,:,0:1]
    coeff2 = coeff[:,:,1:2]
    coeff3 = coeff[:,:,2:3]
    new_verts = (verts[idx1, None, :] * coeff1 +  verts[idx2, None, :] * coeff2 +  verts[idx3, None, :] * coeff3)
    new_verts = new_verts.view(-1,3)
    new_verts = torch.cat([verts, new_verts], dim=0)
    normals = torch.from_numpy(normals).float().to(verts.device)
    new_normals = (normals[idx1, None, :] * coeff1 +  normals[idx2, None, :] * coeff2 +  normals[idx3, None, :] * coeff3)
    new_normals = torch.cat([normals, new_normals.view(-1,3)], dim=0)

    return new_verts, new_normals

if __name__ == '__main__':

    basedir = sys.argv[1]
    subject = sys.argv[2]

    frames = sorted(os.listdir(os.path.join(basedir, subject, 'smpl')))
    # To get the color of SMPL point cloud, we chose 8 source views
    # and use lumigraph blending to fuse the RGB value of each point
    ref_views = [1, 7, 13, 19, 25, 31, 37, 43]

    all_views = list(range(48))
    if subject == 'wuwenyan':
        all_views = list(set(all_views)-set([34, 36]))
    elif (subject == 'dannier' or subject == 'Tichinah_jervier'):
        all_views = list(set(all_views)-set([32]))
    elif subject == 'joseph_matanda':
        all_views = list(set(list(range(48))) - set([39, 40, 42, 43, 44, 45, 46, 47]))

    # select valid reference views
    ref_views = [i for i in ref_views if i in all_views]
    intrins, w2cs, imgs = [], [], []
    annot_path = os.path.join(basedir, subject, 'annots.npy')
    annots = np.load(annot_path, allow_pickle=True).item()
    for i in ref_views:
        idx = all_views.index(i)
        K = np.array(annots['cams']['K'][idx], dtype=np.float32)
        Rt = np.array(annots['cams']['RT'][idx], dtype=np.float32)
        Rt = torch.from_numpy(np.linalg.inv(Rt))
        calib = torch.Tensor([K[0,0],K[1,1],K[0,2],K[1,2]]).float()
        intrins.append(calib)
        w2cs.append(Rt)
    w2cs = torch.stack(w2cs, dim=0).cuda()
    persps = torch.stack(intrins, dim=0).cuda()

    outdir = os.path.join(basedir, subject, 'smpl_color')
    os.makedirs(outdir, exist_ok=True)

    for frame in tqdm(frames):
        verts, faces = load_ply(os.path.join(basedir, subject, 'smpl', frame))
        normals = compute_normal(verts, faces)
        verts = torch.from_numpy(verts).float().cuda()
        verts, normals = pointcloud_upsampling(verts, faces, normals)
        
        imgs = [imageio.imread(os.path.join(basedir, subject, 'image', '%02d'%i, frame[:-4]+'.jpg')) for i in ref_views]
        imgs = np.stack(imgs, axis=0).astype(np.float32)
        imgs = torch.from_numpy(imgs).permute((0,3,1,2)).cuda()
        
        pts = verts.permute(1,0)[None,...].expand([imgs.shape[0], -1, -1])
        xyz = perspective(pts, w2cs, persps)
        xy = xyz[:,:2]
        xy = xy / torch.tensor([[[imgs.shape[-1]],[imgs.shape[-2]]]], \
            dtype = xyz.dtype, device = xyz.device) * 2 - 1
        # get per-view rgb observation
        rgbs = index(imgs, xy)
        # apply lumigraph blending
        rgbs = lumigraph_blending(verts, normals, rgbs, w2cs, 10)

        verts = verts.cpu().numpy()
        rgbs = np.clip(rgbs.cpu().numpy().astype(np.int32), 0, 255)
        save_obj_mesh_with_color(os.path.join(outdir, frame[:-4]+'.obj'), verts, rgbs)