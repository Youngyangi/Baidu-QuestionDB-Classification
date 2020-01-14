from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import numpy as np
import tensorflow as tf

def request_server(input_feature, server_url):
    '''
    用于向TensorFlow Serving服务请求推理结果的函数。
    :param img_resized: 经过预处理的待推理图片数组，numpy array，shape：(h, w, 3)
    :param server_url: TensorFlow Serving的地址加端口，str，如：'0.0.0.0:8500'
    :return: 模型返回的结果数组，numpy array
    '''
    # Request.
    channel = grpc.insecure_channel(server_url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "pb_model"  # 模型名称，启动容器命令的model_name参数
    request.model_spec.signature_name = "serving_default"  # 签名名称，刚才叫你记下来的
    # "input_1"是你导出模型时设置的输入名称，刚才叫你记下来的
    token_ids = input_feature['token_ids']
    seg_ids = input_feature['segment_ids']
    request.inputs["Input-Token"].CopyFrom(tf.make_tensor_proto(token_ids))
    request.inputs["Input-Segment"].CopyFrom(tf.make_tensor_proto(seg_ids))
    response = stub.Predict(request, 5)  # 5 secs timeout
    return np.asarray(response.outputs["classifier"].float_val) # fc2为输出名称，刚才叫你记下来的


# feature = {'token_ids': [101, 2769, 3221, 2207, 3209, 102], 'segment_ids': [0, 0, 0, 0, 0, 0]}
# response = request_server(feature, '0.0.0.0:8500')
# print(response)
