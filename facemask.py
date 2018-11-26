import cv2, time 
# Wait for Cam to load
print("[System]: System call generating code.....")
time.sleep(1.5)
print("[System]: Lighting up the torch......")
vid = cv2.VideoCapture(0)
time.sleep(1)
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')


face_mask = cv2.imread("./flames/flames3.png")
h_mask, w_mask = face_mask.shape[:2]
    

if face_cascade.empty():
	raise IOError ('Unable to load face cascade file, check dir')
scaling_factor = 1.25

while True:
	ret, frame = vid.read()
	frame = cv2.resize(frame, None, fx = scaling_factor, fy = scaling_factor, interpolation = cv2.INTER_AREA)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	face_rects = face_cascade.detectMultiScale(gray, 1.3,5)
	
	
	
	for (x,y,h,w) in face_rects:
	    
		if h > 0 and w > 0:
			h,w = int(1.4*h),int(1.0*w)
			y -= 0.1*h
			y = int(y)
			x = int(x)
		frame_roi = frame[y:y+h, x:x+w]
		face_mask_small = cv2.resize(face_mask, (w,h), interpolation=cv2.INTER_AREA)	

		gray_mask =cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
		ret, mask = cv2.threshold(gray_mask, 180,255,cv2.THRESH_BINARY_INV)
		
		mask_inv = cv2.bitwise_not(mask)
		masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)
		masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask =mask_inv)

		frame[y:y+h, x:x+w] = cv2.add(masked_face, masked_frame)

	cv2.imshow("Magic of Science", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		print("[System]: Slaying the Kraken....")
		time.sleep(1)
		print("[System]: Collapsing Rome.....")
		time.sleep(1)
		break
vid.release()
cv2.destroyAllWindows()

