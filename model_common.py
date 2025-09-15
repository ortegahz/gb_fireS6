import os 
from pathlib import Path
import sys

BaseDir = str(Path(__file__).resolve().parent.parent)
sys.path.append(BaseDir)    

import onnxruntime
import numpy as np



class ONNXModel():
    # def __init__(self, weights,providers=["CUDAExecutionProvider"]):
    def __init__(self, weights,providers=["CPUExecutionProvider"]):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(str(weights),providers=providers)
        self.input_names,self.input_shapes = self.get_input_name(self.onnx_session)
        

        self.output_names= self.get_output_name(self.onnx_session)
        
    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_names = []
        for node in onnx_session.get_outputs():
            output_names.append(node.name)
        return output_names
    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_names = []
        input_shapes = []
        for node in onnx_session.get_inputs():
            input_names.append(node.name)
            input_shapes.append(node.shape)
        return input_names,input_shapes
    
        
    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed
    
    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        input_feed = self.get_input_feed(self.input_names, image_numpy)
        res = self.onnx_session.run(self.output_names, input_feed=input_feed)
        
        return res
    
    @ staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    
if __name__ == "__main__":
    fake_img = np.random.random((1, 3, 480, 640)).astype(np.float32)
    # fake_img = np.random.random((1,3,540,960)).astype(np.float32)
    weights = Path(r"G:\Windows\PycharmProjects1\ultrasmoker\t7best.onnx")
    # fake_img = np.random.random((1,3,540,960)).astype(np.float32)
    # weights = Path("./yolov5s_fire_detection_20240822.onnx")
    model = ONNXModel(weights)
    res = model.forward(fake_img)
    print(len(res))
    for feature in res:
        print(feature.shape)
