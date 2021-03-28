import cv2


def down_sample_till_limit(img, limit = 32):
    images = []
    images.append(img)
    while ((img.shape[1]//2 >= limit) and (img.shape[0]//2 >= limit)):
        img = cv2.pyrDown(img)
        images.append(img)
    return images

def up_sample_till_limit(img, limit = 2048):
    images = []
    images.append(img)
    while ((img.shape[1]*2 <= limit) and (img.shape[0]*2 <= limit)):
        img = cv2.pyrUp(img)
        images.append(img)
    return images

def gaussian_custom_resize(img,limit,ratio,upper_limit = True):
    """
    img - image
    limit - stopping condition
    ratio - >1 to upsacle , <1 to downscale
    upper_limit - True if you want to upsacle the image
    """
    images = []
    images.append(img)
    if upper_limit:    
        while ((img.shape[1]*ratio <= limit) and (img.shape[0]*ratio <= limit)):
            img = cv2.GaussianBlur(img,(3,3))
            img = cv2.resize(img,(0,0),fx=ratio,fy=ratio,interpolation=cv2.INTER_NEAREST)
            # print(str(img.shape[0]) + "  " + str(img.shape[1]))
            images.append(img)
        return images
    else:
        while ((img.shape[1]*ratio >= limit) and (img.shape[0]*ratio >= limit)):
            img = cv2.GaussianBlur(img,(3,3))
            img = cv2.resize(img,(0,0),fx=ratio,fy=ratio,interpolation=cv2.INTER_NEAREST)
            # print(str(img.shape[0]) + "  " + str(img.shape[1]))
            images.append(img)
        return images

def gaussian_custom_resize_pyramid(img,lower_limit = 32,ratio = 0.5):
    images = []
    images.append(img)
    num_images = 0
    while ((img.shape[1]*ratio >= lower_limit) and (img.shape[0]*ratio >= lower_limit)):
        img = cv2.GaussianBlur(img,(3,3))
        img = cv2.resize(img,(0,0),fx=ratio,fy=ratio,interpolation=cv2.INTER_NEAREST)
        # print(str(img.shape[0]) + "  " + str(img.shape[1]))
        images.append(img)
        num_images+=1
    for i in range(num_images):
        img = cv2.GaussianBlur(img,(3,3))
        img = cv2.resize(img,(0,0),fx=1/ratio,fy=1/ratio,interpolation=cv2.INTER_NEAREST)
        # print(str(img.shape[0]) + "  " + str(img.shape[1]))
        images.append(img)
    
    return images

def laplacian_pyramid_with_levels(img,levels=3):
    
    gaussian_pyr = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        gaussian_pyr.append(img)

    laplacian_top = gaussian_pyr[-1]
    laplacian_pyr = [laplacian_top]
    
    for i in range(levels,0,-1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyr[i-1], gaussian_expanded)
        laplacian_pyr.append(laplacian)

    return laplacian_pyr
