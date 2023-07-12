from pylab import *
import torch
from scipy.spatial import cKDTree
import pandas as pd
import reciprocalspaceship as rs


class ImageData(torch.utils.data.Dataset):
    def __init__(self, image_dir, prediction_dir, max_size=4096):
        self.image_files = image_dir
        self.prediction_files = prediction_dir
        self.max_size = max_size

    def __len__(self):
        return len(self.image_files)

    def get_data_set(self, idx):
        image_file = self.image_files[idx]
        prediction_file = self.prediction_files[idx]
        ds = rs.read_precognition(prediction_file)
        ds = ds.reset_index().groupby(["X", "Y"]).first().reset_index() #Needed to remove harmonics
        return ds

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        ds = self.get_data_set(idx)
        pix = imread(image_file)
        x = ds.X.to_numpy('float32')
        y = ds.Y.to_numpy('float32')
        centroids = np.column_stack((x, y))
        
        
        #from dials.array_family import flex
        #Alternative dials version once prediction is fixed
        #prediction_file = "predicted000099.refl"
        #refls = flex.reflection_table.from_file(prediction_file)
        #x,y,_ = np.array(refls['xyzcal.px']).T
                
        xy = np.column_stack((x, y))
        pxy = np.indices(pix.shape).T.reshape((-1, 2))
        tree = cKDTree(xy)
        d,pidx = tree.query(pxy)
        dxy = pxy - centroids[pidx]

        df = pd.DataFrame({
            'counts' : pix.flatten(),
            'dist' : d,
            'x' : pxy[:,0],
            'y' : pxy[:,1],
            'dx' : dxy[:,0],
            'dy' : dxy[:,1],
            'idx' : pidx,
        })
        df['dist_rank'] = df[['idx', 'dist']].groupby('idx').rank(method='first').astype('int') - 1
        m = self.max_size
        n = len(centroids)
        df = df[df.dist_rank < m]

        idx_1,idx_2 = df.idx.to_numpy(),df.dist_rank.to_numpy()
        mask = torch.zeros((n, m), dtype=torch.bool)
        mask[idx_1, idx_2] = True

        counts = torch.zeros((n, m), dtype=torch.float32)
        counts[idx_1, idx_2] = torch.tensor(df.counts.to_numpy('float32'))

        xy = torch.zeros((n, m, 2), dtype=torch.float32)
        xy[idx_1, idx_2, 0] = torch.tensor(df.x.to_numpy('float32'))
        xy[idx_1, idx_2, 1] = torch.tensor(df.y.to_numpy('float32'))

        dxy = torch.zeros((n, m, 2), dtype=torch.float32)
        dxy[idx_1, idx_2, 0] = torch.tensor(df.dx.to_numpy('float32'))
        dxy[idx_1, idx_2, 1] = torch.tensor(df.dy.to_numpy('float32'))

        #Standardized
        idx = torch.clone(xy)
        xy = self.standardize(xy)
        dxy = self.standardize(dxy)

        return idx, xy, dxy, counts, mask

    @staticmethod
    def standardize(x, center=True):
        d = x.shape[-1]
        if center:
            mu = x.reshape((-1, d)).mean(0)  
        else:
            mu = 0.
        sigma = x.reshape((-1, d)).std(0)
        return (x - mu) / sigma

