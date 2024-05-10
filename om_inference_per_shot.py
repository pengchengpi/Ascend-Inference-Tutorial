import os
import cv2
import numpy as np
import acl
from patches_croppping import crop_image_into_patches, reconstruct_image_from_patches
import imageio


NPY_FLOAT32 = 11
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_DEVICE, ACL_HOST = 0, 1
ACL_SUCCESS = 0
min_chn = np.array([123.675, 116.28, 103.53],
                   dtype=np.float32)
var_reci_chn = np.array([0.0171247538316637,
                         0.0175070028011204,
                         0.0174291938997821],
                        dtype=np.float32)

def std_normalize(x):
    mu = np.average(x)
    sigma = np.std(x)
    x = (x - mu) / sigma
    return x

def min_max_normalize(x):
    Min = np.min(x)
    Max = np.max(x)
    x = (x - Min) / (Max - Min)
    return x

class Model_Inference(object):
    def __init__(self, device_id, 
    model_path, model_width, model_height,
    cropping_size = (64, 64),
    overlap_size = (16, 0),img_size=(1024,64)):
        self.device_id = device_id  # int
        self.context = None  # pointer
        self.stream = None

        self.model_width = model_width
        self.model_height = model_height
        self.model_id = None  # pointer
        self.model_path = model_path  # string
        self.model_desc = None  # pointer when using
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffer = None
        self.output_buffer = None
        self.input_buffer_size = None
        self.image_bytes = None
        self.image_name = None
        self.dir = None
        self.image = None
        self.runMode_ = acl.rt.get_run_mode()
        self.bytes_input = None
        self.cropping_size = cropping_size
        self.overlap_size = overlap_size
        self.process_idx_debug = 0
        self.output_patch_list = []
        self.img_size = img_size

    def init_resource(self):
        # init acl resource
        ret = acl.init()
        if ret != ACL_SUCCESS:
            print('acl init failed, errorCode is', ret)

        ret = acl.rt.set_device(self.device_id)
        if ret != ACL_SUCCESS:
            print('set device failed, errorCode is', ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        if ret != ACL_SUCCESS:
            print('create context failed, errorCode is', ret)

        self.stream, ret = acl.rt.create_stream()
        if ret != ACL_SUCCESS:
            print('create stream failed, errorCode is', ret)

        # load model from file
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != ACL_SUCCESS:
            print('load model failed, errorCode is', ret)

        # create description of model
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != ACL_SUCCESS:
            print('get desc failed, errorCode is', ret)

        # create data set of input
        self.input_dataset = acl.mdl.create_dataset()
        input_index = 0
        self.input_buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, input_index)
        self.input_buffer, ret = acl.rt.malloc(self.input_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
        input_data = acl.create_data_buffer(self.input_buffer, self.input_buffer_size)
        self.input_dataset, ret = acl.mdl.add_dataset_buffer(self.input_dataset, input_data)
        if ret != ACL_SUCCESS:
            print('acl.mdl.add_dataset_buffer failed, errorCode is', ret)

        # create data set of output
        self.output_dataset = acl.mdl.create_dataset()
        output_index = 0
        output_buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, output_index)
        self.output_buffer, ret = acl.rt.malloc(output_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
        output_data = acl.create_data_buffer(self.output_buffer, output_buffer_size)
        self.output_dataset, ret = acl.mdl.add_dataset_buffer(self.output_dataset, output_data)
        if ret != ACL_SUCCESS:
            print('acl.mdl.add_dataset_buffer failed, errorCode is', ret)

    def process_input(self, input_path):
        # read image from file by opencv
        # self.dir, self.image_name = os.path.split(input_path)
        # input_path = os.path.abspath(input_path)
        image = np.load(input_path)
        image = image.astype('float32')
        self.image = image

        # HWC to CHW
        image_ = image.transpose([2, 0, 1]).copy()
        self.image_bytes = np.frombuffer(image_.tobytes())

    def image_buffer_handling(self, image_buffer):
        # change N * H * W img buffer into m * C * Hp * Wp
        # C is MFDN channel, m is patch num which can be treated as batch size,
        # Hp, Wp are patch sizes.
        # image_buffer should be N * H * W

        image_patch_sequence = []
        self.image_patch_sequence = image_buffer[0, :, :].shape
        for img in image_buffer:
            img = np.expand_dims(img, axis=-1)
            img = std_normalize(img)
            image_patches = crop_image_into_patches(img,
                                                    self.cropping_size,
                                                    self.overlap_size)
            image_patch_sequence.append(image_patches)
        # change to np array and convert to float32 for Ascend 910/310 devices
        image_patch_sequence = np.asarray(image_patch_sequence).astype(np.float32)
        image_patch_sequence_swapped = image_patch_sequence.swapaxes(0, 1)
        image_patch_sequence_sequeenze = image_patch_sequence_swapped.squeeze()


        input = image_patch_sequence_sequeenze
        self.image_bytes = np.frombuffer(input.tobytes())

    def inference(self):
        # pass image data to input data buffer
        if self.runMode_ == ACL_DEVICE:
            kind = ACL_MEMCPY_DEVICE_TO_DEVICE
        else:
            kind = ACL_MEMCPY_HOST_TO_DEVICE
        if "bytes_to_ptr" in dir(acl.util):
            bytes_data = self.image_bytes.tobytes()
            self.bytes_input = bytes_data
            ptr = acl.util.bytes_to_ptr(bytes_data)
        else:
            ptr = acl.util.numpy_to_ptr(self.image_bytes)
        ret = acl.rt.memcpy(self.input_buffer,
                            self.input_buffer_size,
                            ptr,
                            self.input_buffer_size,
                            kind)
        if ret != ACL_SUCCESS:
            print('memcpy failed, errorCode is', ret)

        # inference
        ret = acl.mdl.execute(self.model_id,
                              self.input_dataset,
                              self.output_dataset)
        if ret != ACL_SUCCESS:
            print('execute failed, errorCode is', ret)

    def get_result(self):
        # get result from output data set
        output_index = 0
        output_data_buffer = acl.mdl.get_dataset_buffer(self.output_dataset, output_index)
        output_data_buffer_addr = acl.get_data_buffer_addr(output_data_buffer)
        output_data_size = acl.get_data_buffer_size(output_data_buffer)
        ptr, ret = acl.rt.malloc_host(output_data_size)

        # copy device output data to host
        if self.runMode_ == ACL_DEVICE:
            kind = ACL_MEMCPY_DEVICE_TO_HOST
        else:
            kind = ACL_MEMCPY_HOST_TO_HOST
        ret = acl.rt.memcpy(ptr,
                            output_data_size,
                            output_data_buffer_addr,
                            output_data_size,
                            ACL_MEMCPY_HOST_TO_DEVICE)
        if ret != ACL_SUCCESS:
            print('memcpy failed, errorCode is', ret)

        index = 0
        dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, index)

        if ret != ACL_SUCCESS:
            print('get output dims failed, errorCode is', ret)
        out_dim = dims['dims']

        if "ptr_to_bytes" in dir(acl.util):
            bytes_data = acl.util.ptr_to_bytes(ptr, output_data_size)
            data = np.frombuffer(bytes_data, dtype=np.float32).reshape(out_dim)
        else:
            data = acl.util.ptr_to_numpy(ptr, out_dim, NPY_FLOAT32)

        self.output_patch_list = data

    def reconstruct_patch_2_image(self, output_patches):
        img_re = reconstruct_image_from_patches(output_patches,
                                                self.img_size,
                                                overlap=self.overlap_size)
        return img_re

    def release_resource(self):
        # release resource includes acl resource, data set and unload model
        acl.rt.free(self.input_buffer)
        acl.mdl.destroy_dataset(self.input_dataset)

        acl.rt.free(self.output_buffer)
        acl.mdl.destroy_dataset(self.output_dataset)
        ret = acl.mdl.unload(self.model_id)
        if ret != ACL_SUCCESS:
            print('unload model failed, errorCode is', ret)

        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        if self.stream:
            ret = acl.rt.destroy_stream(self.stream)
            if ret != ACL_SUCCESS:
                print('destroy stream failed, errorCode is', ret)
            self.stream = None

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            if ret != ACL_SUCCESS:
                print('destroy context failed, errorCode is', ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        if ret != ACL_SUCCESS:
            print('reset device failed, errorCode is', ret)

        ret = acl.finalize()
        if ret != ACL_SUCCESS:
            print('finalize failed, errorCode is', ret)


if __name__ == '__main__':
    device = 0
    model_width = 64
    model_height = 64
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = './FastDVDNet.om'
    if not os.path.exists(model_path):
        raise Exception("the model is not exist")


    # load images' inputs
    image_paths = ["./test_imgs/271_rd_abs.npy",
                   "./test_imgs/272_rd_abs.npy",
                   "./test_imgs/273_rd_abs.npy",
                   "./test_imgs/274_rd_abs.npy",
                   "./test_imgs/275_rd_abs.npy"]

    # load imgs
    imgs = []
    for path in image_paths:
        imgs.append(np.load(path))
    imgs = np.asarray(imgs)


    path = '/home/user/ppc/samples/inference/modelInference/sampleResnetQuickStart/python/img_15dB_patch.npy'

    # inference
    net = Model_Inference(device, model_path, model_width, model_height)
    net.init_resource()


    net.image_buffer_handling(imgs)
    net.inference()
    net.get_result()

    # reconstruct image patch2img
    infer_results = np.asarray(net.output_patch_list)
    infer_results = infer_results.squeeze()
    infer_results = np.expand_dims(infer_results, axis=-1)
    img_re = net.reconstruct_patch_2_image(infer_results)
    
    # post processing for saving img
    img_re = min_max_normalize(img_re)
    img_re = img_re / img_re.max()
    img_re = 255 * img_re
    img_re = img_re.astype(np.uint8)
    imageio.imwrite('./test_output_per_shot.jpg', img_re[:, :, 0])

    print("*****run finish******")
    net.release_resource()