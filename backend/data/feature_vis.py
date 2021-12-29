import numpy as np
from PIL import Image
import jittor as jt
from jittor import nn
from jittor import transform
from jittor.lr_scheduler import CosineAnnealingLR

from data.jimm import resnet26


from data.feature_utils import *

model_dict_path = '/home/fengyuan/JittorModels/trained-models/restnet-14-0.98.pkl'
img_path = '/home/fengyuan/JittorModels/trainingSet/trainingSet/0/img_1.jpg'

def get_train_transforms():
    return transform.Compose([
        transform.RandomCropAndResize((448, 448)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def get_valid_transforms():
    return transform.Compose([
        transform.Resize(448),
        # transform.CenterCrop(448),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


class Extractor():
    """
        Extract grad and output of a model
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.model_output = None

        self.conv_output = []
        self.relu_output = []
        self.conv_grads_out = []
        self.relu_grads_out = []
        self.conv_grads_in = []
       
    def remove_hooks(self):
        for module_pos, module in enumerate(unpack_children(self.model)):
            if isinstance(module, jt.nn.ReLU):
                module.remove_forward_hook()
                module.remove_backward_hook()
            elif isinstance(module, jt.nn.Conv2d):
                module.remove_forward_hook()
                module.remove_backward_hook()

    def clear_cache(self):
        self.conv_output = []
        self.relu_output = []
        self.conv_grads_out = []
        self.relu_grads_out = []
        self.conv_grads_in = []

    def register_hooks(self, type="cam"):
        self.remove_hooks()
        self.clear_cache()
        if type=="cam":
            self.register_cam_hooks()
        elif type=="gbp":
            self.register_gbp_hooks()
        elif type=="vbp":
            self.register_vbp_hooks()
        else:
            print("wrong hook type")

    def register_cam_hooks(self):
        def relu_forward_hook(m, t_in, t_out):
            self.relu_output.append(t_out)
        def relu_backward_hook(m, g_in, g_out):
            self.relu_grads_out.append(g_out[0])
        def conv_forward_hook(m, t_in, t_out):
            self.conv_output.append(t_out)
        def conv_backward_hook(m, g_in, g_out):
            self.conv_grads_out.append(g_out[0])
        
        for module_pos, module in enumerate(unpack_children(self.model)):
            if isinstance(module, jt.nn.ReLU):
                module.register_forward_hook(relu_forward_hook)
                module.register_backward_hook(relu_backward_hook)
            elif isinstance(module, jt.nn.Conv2d):
                module.register_forward_hook(conv_forward_hook)
                module.register_backward_hook(conv_backward_hook)

    def register_vbp_hooks(self):

        def conv_backward_hook(m, g_in, g_out):
            self.conv_grads_in.append(g_in[0])
        
        def relu_backward_hook(m, g_in, g_out):
            g_in[0].data
        
        for module_pos, module in enumerate(unpack_children(self.model)):
            if isinstance(module, jt.nn.ReLU):
                module.register_backward_hook(relu_backward_hook)
            elif isinstance(module, jt.nn.Conv2d):
                module.register_backward_hook(conv_backward_hook)

    def register_gbp_hooks(self):
        
        def relu_backward_hook(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            grad_in[0].data
            corresponding_forward_output = self.relu_output[-1]
            corresponding_forward_output.data[corresponding_forward_output.data > 0] = 1
            modified_grad_out = corresponding_forward_output * jt.clamp(grad_in[0], min_v=0.0)
            del self.relu_output[-1]
            return (modified_grad_out,)

        def relu_forward_hook(m, t_in, t_out):
            self.relu_output.append(t_out)

        def conv_backward_hook(m, g_in, g_out):
            self.conv_grads_in.append(g_in[0])
        
        for module_pos, module in enumerate(unpack_children(self.model)):
            if isinstance(module, jt.nn.ReLU):
                module.register_forward_hook(relu_forward_hook)
                module.register_backward_hook(relu_backward_hook)
            elif isinstance(module, jt.nn.Conv2d):
                module.register_backward_hook(conv_backward_hook)

    def forward_and_backward(self, model_input, target_class):
        model_input.start_grad()

        self.model_output = self.model(model_input)

        if not target_class:
            target_class = np.argmax(self.model_output.data)
        one_hot_output = jt.float32([[0 for i in range(self.model_output.size()[-1])]])
        one_hot_output.data[0][target_class] = 1

        criterion = nn.CrossEntropyLoss()
        loss = criterion(self.model_output, one_hot_output)

        model_input_grad = jt.grad(loss, model_input)
        model_input_grad.sync()

    def generate_gradients(self, input_image, target_class):
        # grad at first conv 
        self.clear_cache()
        self.forward_and_backward(input_image, target_class)
        gradients = self.conv_grads_in[-1]
        self.clear_cache()
        return gradients

    # ------for integrated grad------------
    def generate_images_on_linear_path(self, input_image, steps):
        step_list = np.arange(steps+1)/steps
        xbar_list = [input_image*step for step in step_list]
        return xbar_list

    def generate_integrated_gradients(self, model_input, target_class, steps=10):
        xbar_list = self.generate_images_on_linear_path(model_input, steps)
        integrated_grads = np.zeros(model_input.size())
        for xbar_image in xbar_list:
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            integrated_grads = integrated_grads + single_integrated_grad/steps
        return integrated_grads[0]


class FeatureVis():
    """
        Algorithms for feature visualization 
            (bp)          vanilla_bp, guided_bp  
            (cam)         grad_cam, layer_cam, guided_grad_cam
            (grad)        integrated_gradients
            (gradximg)    [(gbp/integ)_]grad_times_image, 
    """
    def __init__(self, model=None):
        self.model = model
        self.model.eval()

        self.transform = get_valid_transforms()

        self.ori_img = get_img(img_path)
        self.model_input = self.transform(self.ori_img)
        self.model_input = self.model_input.reshape((1,3,448,448))
        self.model_input = jt.array(self.model_input)

        self.target_class = extract_target_class(img_path)

    def get_feature_vis(self, model_input=None, target_class=None, method='vanilla_bp'):
        """
            model_input (numpy, [w, h, 3])
            return numpy([w, h, 3])
        """
        if model_input is not None:
            if model_input.shape[2]==3: # if shape (w, h, 3)
                img_input = model_input.transpose((2,0,1))
            img_input = self.transform(model_input)
            w, h = img_input.shape[1], img_input.shape[2]
            img_input = img_input.reshape((1, 3, w, h))
            img_input = jt.array(img_input)
            
        print(img_input.shape, target_class)
        if not 'cam' in method:
            grads = eval("self.%s" % method)(img_input=img_input, target_class=target_class)
        else:
            ori_img = Image.fromarray(np.uint8(model_input))
            grads = eval("self.%s" % method)(img_input=img_input, target_class=target_class, ori_img=ori_img)
        
        grads = grads.transpose((1,2,0)) # change to (w, h, 3)
        return grads

    def vanilla_bp(self, img_input, target_class, file_name_to_export='vanilla_bp', save=False):
        extractor = Extractor(self.model)
        extractor.register_hooks(type="vbp")
        first_conv_grad = extractor.generate_gradients(img_input, target_class)[0].data
        vanilla_grads = first_conv_grad

        if save:
            save_gradient_images(vanilla_grads, file_name_to_export + '_Guided_BP_color')
            grayscale_guided_grads = convert_to_grayscale(vanilla_grads)
            save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
            pos_sal, neg_sal = get_positive_negative_saliency(vanilla_grads)
            save_gradient_images(pos_sal, file_name_to_export + '_p_sal')
            save_gradient_images(neg_sal, file_name_to_export + '_n_sal')
        
        # grayscale_guided_grads = convert_to_grayscale(vanilla_grads)
        # vanilla_grads = grayscale_guided_grads

        print('vanilla_bp', type(vanilla_grads), vanilla_grads.shape)
        
        vanilla_grads = normalize(vanilla_grads)

        return vanilla_grads

    def guided_bp(self, img_input, target_class, file_name_to_export='guided_bp', save=False):
        extractor = Extractor(self.model)
        extractor.register_hooks("gbp")
        first_conv_grad = extractor.generate_gradients(img_input, target_class)[0].data
        guided_grads = first_conv_grad

        if save:
            save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
            grayscale_guided_grads = convert_to_grayscale(guided_grads)
            save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
            pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
            save_gradient_images(pos_sal, file_name_to_export + '_p_sal')
            save_gradient_images(neg_sal, file_name_to_export + '_n_sal')

        print('guided_bp', type(guided_grads), guided_grads.shape)
        return normalize(guided_grads)

    def integrated_gradients(self, img_input, target_class, steps=10, file_name_to_export='integrated_gradients', save=False):
        extractor = Extractor(self.model)
        extractor.register_hooks("vbp")
        integrated_grads = extractor.generate_integrated_gradients(img_input, target_class, steps)
        integrated_grads = integrated_grads.data
        
        if save:
            grayscale_integrated_grads = convert_to_grayscale(integrated_grads)
            save_gradient_images(grayscale_integrated_grads, file_name_to_export + '_Integrated_G_gray')

        print('integrated_gradients', type(integrated_grads), integrated_grads.shape)
        grayscale_integrated_grads = convert_to_grayscale(integrated_grads)
        return normalize(grayscale_integrated_grads)

    def grad_times_image(self, img_input, target_class, file_name_to_export='gradximg', save=False):
        extractor = Extractor(self.model)
        extractor.register_hooks("vbp")
        first_conv_grad = extractor.generate_gradients(img_input, target_class)[0].data
        vanilla_grads = first_conv_grad
        grad_times_image = vanilla_grads * img_input.numpy()[0]

        if save:
            grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
            save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_grad_times_image_gray')

        print('grad_times_image', type(grad_times_image), grad_times_image.shape)
        grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
        return normalize(grayscale_vanilla_grads)

    def gbp_grad_times_image(self, img_input, target_class, file_name_to_export='gbp_gradximg', save=False):
        extractor = Extractor(self.model)
        extractor.register_hooks("gbp")
        first_conv_grad = extractor.generate_gradients(img_input, target_class)[0].data
        gbp_grads = first_conv_grad
        gbp_grad_times_image = gbp_grads * img_input.numpy()[0]

        if save:
            grayscale_vanilla_grads = convert_to_grayscale(gbp_grad_times_image)
            save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_grad_times_image_gray')

        print('gbp_grad_times_image', type(gbp_grad_times_image), gbp_grad_times_image.shape)
        grayscale_vanilla_grads = convert_to_grayscale(gbp_grad_times_image)
        return normalize(grayscale_vanilla_grads)

    def integ_grad_times_image(self, img_input, target_class, file_name_to_export='integ_gradximg', save=False):
        extractor = Extractor(self.model)
        extractor.register_hooks("vbp")
        integrated_grads = extractor.generate_integrated_gradients(img_input, target_class).data
        integ_grad_times_image = integrated_grads * img_input.numpy()[0]

        if save:
            grayscale_vanilla_grads = convert_to_grayscale(integ_grad_times_image)
            save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_grad_times_image_gray')

        print('integ_grad_times_image', type(integ_grad_times_image), integ_grad_times_image.shape)
        grayscale_vanilla_grads = convert_to_grayscale(integ_grad_times_image)
        return normalize(grayscale_vanilla_grads)

    def grad_cam(self, img_input, target_class, ori_img=None, file_name_to_export='grad_cam', save=False, cam_size=None, return_cam=False):
        extractor = Extractor(self.model)
        extractor.register_hooks("cam")
        extractor.forward_and_backward(img_input, target_class)
        last_conv_grad = extractor.conv_grads_out[0][0].data
        last_conv_output = extractor.conv_output[-1][0].data
        weights = np.mean(last_conv_grad, axis=(1, 2))
        target = last_conv_output
        cam = np.ones(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)
        if not cam_size:
            cam_size = ori_img.size
        cam = np.uint8(Image.fromarray(cam).resize(cam_size, Image.ANTIALIAS))/255

        if save:
            save_class_activation_images(ori_img, cam, file_name_to_export)

        if return_cam:
            return cam
        else:
            heat_map_on_image = get_class_activation_image(ori_img, cam)
            print('grad_cam', type(heat_map_on_image), heat_map_on_image.shape)
            return heat_map_on_image

    def guided_grad_cam(self, img_input, target_class=None, ori_img=None, file_name_to_export='guided_grad_cam', save=False):
        cam = self.grad_cam(img_input, save=False, cam_size=tuple(img_input.shape[2:]), return_cam=True)
        guided_grads = self.guided_bp(img_input, save=False)
        cam_gb = np.multiply(cam, guided_grads)

        if save:
            save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
            grayscale_cam_gb = convert_to_grayscale(cam_gb)
            save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')

        return normalize(cam_gb)

    def layer_cam(self, img_input, target_class, ori_img, file_name_to_export='layer_cam', save=False, return_cam=False):
        extractor = Extractor(self.model)
        extractor.register_hooks("cam")
        extractor.forward_and_backward(img_input, target_class)
        last_conv_grad = extractor.conv_grads_out[0][0].data
        last_conv_output = extractor.conv_output[-1][0].data
        weights = last_conv_grad
        target = last_conv_output
        weights[weights < 0] = 0
        cam = np.sum(weights * target, axis=0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)
        cam = np.uint8(Image.fromarray(cam).resize(ori_img.size, Image.ANTIALIAS))/255

        if save:
            save_class_activation_images(ori_img, cam, file_name_to_export)

        if return_cam:
            return cam
        else:
            heat_map_on_image = get_class_activation_image(ori_img, cam)
            print('layer_cam', type(heat_map_on_image), heat_map_on_image.shape)
            return heat_map_on_image

if __name__=="__main__":
    model = resnet26(pretrained=False, num_classes=10)
    model.load_state_dict(jt.load(model_dict_path))
    featurevis = FeatureVis(model)
    vis = featurevis.get_feature_vis()
    print(vis.shape)



        