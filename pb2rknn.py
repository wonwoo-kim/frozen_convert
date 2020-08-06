from rknn.api import RKNN

rknn = RKNN()

print('--> loading model')

ret = rknn.load_tensorflow(tf_pb='./preds_frozen0.pb',
		inputs=['start'],
		outputs=['preds'],
		input_size_list=[[1,256,256,3]])

if ret !=0:
	print('load failed!')
	exit(ret)


print('done')

print('--> building model')

ret=rknn.build(do_quantization=False)

if ret !=0:
	print('build failed')
	exit(ret)

print('done')


print('-->export rknn model')

ret = rknn.export_rknn('./preds_model.rknn')

if ret !=0:
	print('export failed')
	exit(ret)

print('done')

