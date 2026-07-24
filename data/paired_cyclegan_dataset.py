import os
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image


class PairedCycleganDataset(BaseDataset):
    """
    This dataset class can load paired datasets for CycleGAN.
    It expects two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    Images in domain A and B must be perfectly paired by file sorting (e.g. identical filenames).

    Unlike UnalignedDataset, this dataset perfectly pairs the images and applies
    the exact same randomized transformations (crops, flips) to both A and B to preserve alignment.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
        """
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size] # perfectly aligned pair

        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")

        # Get synchronized random transform parameters (same crop and flip for both)
        transform_params = get_params(self.opt, A_img.size)

        transform_A = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        transform_B = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # apply image transformation
        A = transform_A(A_img)
        B = transform_B(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
