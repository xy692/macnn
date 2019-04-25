import caffe
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import tensorflow as tf


#this function is to get the location of bird which is the high conv response area
def bird_box(net):
	mask = net.blobs['upsample'].data.reshape(448,448)
	mask_max = np.max(mask.flat)
	t = mask_max * 0.1
	t1 = np.max(mask,axis = 0)
	for j in range(448):
		if t1[j]>t:
			left = j
			break
	for j in range(447,-1,-1):
		if t1[j]>t:
			right = j
			break
	t2 = np.max(mask,axis = 1)
	for j in range(448):
		if t2[j]>t:
			up = j
			break
	for j in range(447,-1,-1):
		if t2[j]>t:
			down = j
			break
	x = (left + right)/2
	y = (up + down)/2
	l = np.maximum(right - left,down - up)/2
	return x,y,l

#this function is to get four part locations which is the output of our designed part network
def part_box(net):
	part1 = np.argmax(net.blobs['mask_1_4'].data.reshape(28,28))
	part2 = np.argmax(net.blobs['mask_2_4'].data.reshape(28,28))
	part3 = np.argmax(net.blobs['mask_3_4'].data.reshape(28,28))
	part4 = np.argmax(net.blobs['mask_4_4'].data.reshape(28,28))

	return part1 % 28, part1 / 28, part2 % 28, part2 / 28, part3 % 28, part3 / 28, part4 % 28, part4 / 28


# convert gray to rgb image
def gray2rgb(rgb,imggray):
   B = imggray
 
   grayRgb = np.zeros((rgb.shape))
   grayRgb[:, :, 2] = B
   grayRgb[:, :, 0] = B
   grayRgb[:, :, 1] = B
 
   return grayRgb

#input the original image and bird location, crop the bird area and resize to 224*224
def get_bird(img,x,y,l):
	[n,m,_]=img.shape
	if m>n:
		islong=1
	else:
		islong=0
	if islong==0:
		x =  x*m/448
		y =  y*m/448 + (n - m) / 2
		l =  l*m/448
	else:
		x =  x*n/448 + (m - n) / 2
		y =  y*n/448
		l =  l*n/448
	box = (np.maximum(0,np.int(x - l)), np.maximum(0,np.int(y - l)),
		np.minimum(m,np.int(x + l)), np.minimum(n,np.int(y + l)))
	return cv2.resize(img[box[1]:box[3],box[0]:box[2],:],(224,224))

#input the original image and part locations, crop the four parts and resize to 224*224
def get_part(img,parts):
	img_parts = [[] for i in range(4)]
	[n,m,_]=img.shape
	if m>n:
		islong=1
	else:
		islong=0
	for i in range(4):
		if islong==0:
			parts[i][0] =  parts[i][0]*m/28+8
			parts[i][1] =  parts[i][1]*m/28+8 + (n - m) / 2
			l =  64*m/448
		else:
			parts[i][0] =  parts[i][0]*n/28+8 + (m - n) / 2
			parts[i][1] =  parts[i][1]*n/28+8
			l =  64*n/448
		box = (np.maximum(0,np.int(parts[i][0] - l)), np.maximum(0,np.int(parts[i][1] - l)),
		 np.minimum(m,np.int(parts[i][0] + l)), np.minimum(n,np.int(parts[i][1] + l)))
		img_parts[i] = cv2.resize(img[box[1]:box[3],box[0]:box[2],:],(224,224))
	return img_parts

#translate an img to fit the input of a network
def data_trans(img, shape):
	mu = np.array([109.973,127.338,123.883])
	transformer = caffe.io.Transformer({'data': shape})
	transformer.set_transpose('data', (2,0,1))  
	transformer.set_mean('data', mu)			
	transformer.set_raw_scale('data', 255)	 
	# transformer.set_channel_swap('data', (2,1,0)) 
	transformed_image = transformer.preprocess('data', img)
	return transformed_image

#crop the centor part from 448*n (keep image ratio) to 448*448
def crop_centor(img):
	[n,m,_]=img.shape
	if m>n:
		m = m*448/n
		n = 448
	else:
		n = n*448/m
		m=448
	return data_trans(cv2.resize(img,(m,n))/255.0,(1,3,n,m))[:,(n-448)/2:(n+448)/2,(m-448)/2:(m+448)/2]


#crop the centor part from 256*n (keep image ratio) to 224*224
def crop_lit_centor(img):
	[n,m,_]=img.shape
	if m>n:
		m = m*256/n
		n = 256
	else:
		n = n*256/m
		m=256
        #return data_trans(cv2.resize(img,(m,n))/255.0,(1,3,n,m))[:,(n-224)/2:(n+224)/2,(m-224)/2:(m+224)/2]
        return cv2.resize(img,(m,n))[(n-224)/2:(n+224)/2,(m-224)/2:(m+224)/2,:]

def get_att(net):
	part_array=[net.blobs['mask_1_4'].data,\
	net.blobs['mask_2_4'].data,\
	net.blobs['mask_3_4'].data,\
	net.blobs['mask_4_4'].data]

	# mask_array=[net.blobs['mask_1_4'].data.reshape(28,28),\
	# net.blobs['mask_2_4'].data.reshape(28,28),\
	# net.blobs['mask_3_4'].data.reshape(28,28),\
	# net.blobs['mask_4_4'].data.reshape(28,28)]

	for i in range(4):
		part1_attention = part_array[i]
		att_max = np.max(part1_attention)
		att_min = att_max - 0.25

		for j in range(len(part1_attention[0])):
			if part1_attention[0][j] >= att_min:
				part1_attention[0][j]=255
			else:
				part1_attention[0][j]=0
		
		part_array[i]=part1_attention[0].reshape(28,28)
		part1_gray=part_array[i].astype(int)
		part_resize=np.zeros((28,28,3))
		part_resize=gray2rgb(part_resize,part1_gray)
		part1_gray=tf.image.resize_images(part_resize,(224,224),method=1)
		# part1_attention = mask_array[i]
		# convert = part1_attention*255
		# part1_gray=np.floor(convert)
		# part1_gray=part1_attention.astype(int)
		# part1_gray=part1_gray.astype(int)
		# plt.subplot(2,2,i+1)
		# plt.imshow(part1_gray,cmap='gray',interpolation='bicubic')
		# with tf.Session() as sess:
		# 	sess.run(tf.global_variables_initializer())
		# 	image = sess.run(part1_gray)
		sess=tf.Session()
		sess.run(tf.global_variables_initializer())
		image = sess.run(part1_gray)
		image = image.astype(np.uint8)
		plt.imshow(image)
		plt.show()

def get_colored_img(img,net):
	part_array=[net.blobs['mask_1_4'].data,\
	net.blobs['mask_2_4'].data,\
	net.blobs['mask_3_4'].data,\
	net.blobs['mask_4_4'].data]
	image=tf.image.resize_images(img,(448,448),method=1)

	for i in range(4):
		part1_attention = part_array[i]
		att_max = np.max(part1_attention)
		att_min = att_max - 0.15

		for j in range(len(part1_attention[0])):
			if part1_attention[0][j] >= att_min:
				part1_attention[0][j]=255
			else:
				part1_attention[0][j]=0
		
		part_array[i]=part1_attention[0].reshape(28,28)
		part1_gray=part_array[i].astype(int)
		part_resize=np.zeros((28,28,3))
		part_resize=gray2rgb(part_resize,part1_gray)
		part1_gray=tf.image.resize_images(part_resize,(448,448),method=1)

		sess = tf.Session()
		bool_mat= sess.run(tf.equal(part1_gray,255))
		part1_gray=sess.run(part1_gray)

		if i==0:
			part1_gray[:,:,1:][part1_gray[:,:,1:]==255]=0
		if i==1:
			part1_gray[:,:,2][part1_gray[:,:,2]==255]=0
		if i==2:
			part1_gray[:,:,:2][part1_gray[:,:,:2]==255]=0
		if i==3:
			part1_gray[part1_gray==255]=120

		part1_gray=part1_gray.astype(np.uint8)
		image = sess.run(tf.where(bool_mat,part1_gray,image))
		
	plt.imshow(image)
	plt.show()

abs_address='/home/xingyunyang/Documents/workspace/release/'
prefix='CUB_200_2011/images/'
ls = 3597
caffe.set_mode_gpu()
caffe.set_device(0)
# mu = array([109.973,127.338,123.883])
model_weights =abs_address+'model/bird_part.caffemodel'
model_def =abs_address+'deploy/bird_part_deploy.prototxt'
p_net = caffe.Net(model_def,model_weights,caffe.TEST)

image_list = open(abs_address+'bird_data/test_list.txt').readlines()

for i in range(ls):	
	print i
	if i>-1:
		img = cv2.imread(abs_address+'bird_data/' + prefix +image_list[i].split(' ')[0])
                #print image_list[i].split(' ')[0]
		if img.ndim<3:
			img = np.transpose(np.array([img,img,img]),(1,2,0))
		[n,m,_]=img.shape
		data = crop_centor(img)
		lit_data = crop_lit_centor(cv2.resize(img,(256,256)))
		p_net.blobs['data'].data[...] = data
		p_out = p_net.forward()
		x,y,l = bird_box(p_net)
		get_colored_img(img,p_net)

		part_boxs = part_box(p_net)
		img_bird = get_bird(img,x,y,l)
		img_part = get_part(img,np.array(part_boxs).reshape((4,2)))
                # Save Image-part
                class_name = os.path.basename(os.path.dirname(image_list[i].split(' ')[0]))
                image_name = os.path.basename(image_list[i].split(' ')[0])
                lit_data = np.array(lit_data)
                print lit_data.shape
                if not os.path.exists('./data'):
                    os.mkdir('./data')

                for i in range(1,7):
                    if not os.path.exists('./data/'+str(i)):
                        os.mkdir('./data/'+str(i))

                if not os.path.exists('./data/1/'+class_name+'/'):
                    os.mkdir('./data/1/'+class_name+'/')
                cv2.imwrite('./data/1/'+class_name+'/'+image_name, lit_data) 
                
                if not os.path.exists('./data/2/'+class_name+'/'):
                    os.mkdir('./data/2/'+class_name+'/')
                cv2.imwrite('./data/2/'+class_name+'/'+image_name, img_bird)

                if not os.path.exists('./data/3/'+class_name+'/'):
                    os.mkdir('./data/3/'+class_name+'/')
                cv2.imwrite('./data/3/'+class_name+'/'+image_name, img_part[0])

                if not os.path.exists('./data/4/'+class_name+'/'):
                    os.mkdir('./data/4/'+class_name+'/')
                cv2.imwrite('./data/4/'+class_name+'/'+image_name, img_part[1])

                if not os.path.exists('./data/5/'+class_name+'/'):
                    os.mkdir('./data/5/'+class_name+'/')
                cv2.imwrite('./data/5/'+class_name+'/'+image_name, img_part[2])

                if not os.path.exists('./data/6/'+class_name+'/'):
                    os.mkdir('./data/6/'+class_name+'/')
                cv2.imwrite('./data/6/'+class_name+'/'+image_name, img_part[3])
                
