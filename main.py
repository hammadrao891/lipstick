from flask import Flask,Response, request, jsonify
import base64
import json
import cv2
import dlib
import faceBlendCommon as face
import numpy as np

def blend_lips_on_face(image_path):
    # Load Image
    im = cv2.imread(image_path)
    
    # Detect face landmarks
    PREDICTOR_PATH = r"shape_predictor_68_face_landmarks.dat"
    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)
    landmarks = face.getLandmarks(faceDetector, landmarkDetector, im)
    
    # Create a mask for the lips
    lipsPoints = landmarks[48:67]
    mask = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(lipsPoints), (1.0, 1.0, 1.0))
    mask = 255 * np.uint8(mask)
    
    # Apply close operation to improve mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
    
    # Blur the mask to obtain a natural result
    mask = cv2.GaussianBlur(mask, (15, 15), cv2.BORDER_DEFAULT)
    
    # Calculate the inverse mask
    inverseMask = cv2.bitwise_not(mask)
    
    # Convert masks to float to perform blending
    mask = mask.astype(float) / 255
    inverseMask = inverseMask.astype(float) / 255
    
    # Apply color mapping for the lips
    lips = cv2.applyColorMap(im, cv2.COLORMAP_INFERNO)
    
    # Convert lips and face to the 0-1 range
    lips = lips.astype(float) / 255
    ladyFace = im.astype(float) / 255
    
    # Multiply lips and face by the masks
    justLips = cv2.multiply(mask, lips)
    justFace = cv2.multiply(inverseMask, ladyFace)
    
    # Add face and lips
    result = justFace + justLips
    
    return result


app = Flask(__name__)

@app.route('/lip', methods=['POST'])
def kk_route():
    if 'image' not in request.files:
     return jsonify({'error': 'No image provided'})

    image = request.files['image']
    image_path = "uploaded_image.jpg"  # You can use a different file name or path

    image.save(image_path)

    result_image = blend_lips_on_face(image_path)
    _, img_encoded = cv2.imencode('.png', result_image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')  # Convert bytes to base64 string
    cv2.imshow("", result_image)
    cv2.waitKey(0)
if __name__ == '__main__':
    app.run(debug=True)
