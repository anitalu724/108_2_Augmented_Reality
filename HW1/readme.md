# HW1 ReadMe
###### tags: `B06901011`, `Augmented Reality`

#### Implement
```shell
$ python3 hw1.py [img_path]
```

#### Specification
* **Read Image**
  * Use ==OpenCV== to read the input image.
  * And then, use lower and upper boundary to get A, B, C's XY coordinates.
  
    ```python
    img_rgb = (cv2.imread(filename))[:,:,::-1]
    lower_A , lower_B, lower_C = np.array([254, -1, -1]), np.array([254, 99, -1]), np.array([-1, -1, 254])
    upper_A , upper_B, upper_C = np.array([256, 1,1]), np.array([256, 101,1]), np.array([0, 0,256])
    mask_A,mask_B,mask_C = cv2.inRange(img_rgb, lower_A, upper_A),cv2.inRange(img_rgb, lower_B, upper_B),cv2.inRange(img_rgb, lower_C, upper_C)
    outputA,outputB,outputC = cv2.bitwise_and(img_rgb, img_rgb, mask = mask_A),cv2.bitwise_and(img_rgb, img_rgb, mask = mask_B),cv2.bitwise_and(img_rgb, img_rgb, mask = mask_C)
    rA, rB, rC = np.where(outputA == 255),np.where(outputB == 255),np.where(outputC == 255)
    ```

* **Coordinate Transformation**
  * Because the original coordinates are mirror image symmetry, so I try to transform the coordinates to those we are familiar with. (Center = A) 
  * Therefore, it is more convenient for me to debug.
  
* **Pinpole Imaging Theory**
  * Use Pinpole Imaging Theory to get the default z value

* **Get O default x, y value**
  * Use the similarity of vectors to get the default value of O

* **Church Algorithm**
  * Use derivative to modified the O's coordinates
  * diff must < 10^-5^ (unit: radian)

