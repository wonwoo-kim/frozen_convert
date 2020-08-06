from rknn.api import RKNN

rknn = RKNN(verbose=True)

print('--> loading model')

ret = rknn.load_tflite(model='./start_0.tflite')

if ret !=0:
	print('load failed!')
	exit(ret)

print('done')


print('--> building model')
ret = rknn.build(do_quantization=False)

if ret !=0:
	print('build failed!')
	exit(ret)

print('done')


print('--> export rknn model')

ret = rknn.export_rknn('./start_0.rknn')

if ret !=0:
	print('export failed!')
	exit(ret)

print('done')

