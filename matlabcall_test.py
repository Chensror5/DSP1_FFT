import numpy as np
import matlab.engine
eng = matlab.engine.start_matlab()


print("Getting the images from matlab...")
dist_image_1_m,dist_image_2_m,noised_image_m,imp_resp_image_m = eng.extract_img_gen('CHEN','YOAV', nargout = 4)
print("Getting the images from matlab - Done")
#redandent, b, c = eng.extract_img_gen(1,2, nargout = 3)
#print("hello1")
#a = np.array(redandent[0])
dist_image_1 = np.array(dist_image_1_m)
dist_image_2 = np.array(dist_image_2_m)
noised_image = np.array(noised_image_m)
imp_resp_image = np.array(imp_resp_image_m)

