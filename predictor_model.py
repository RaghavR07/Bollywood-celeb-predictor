import os
import pickle
from tqdm import tqdm
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image
#
actors=os.listdir('bollywood_celeb_faces')
print(actors)

filenames=[]

for actor in actors:
    for file in os.listdir(os.path.join('bollywood_celeb_faces',actor)):
        filenames.append(os.path.join('bollywood_celeb_faces',actor,file))
#
#
# print(filenames)
# print(len(filenames))
#
# pickle.dump(filenames,open('filenames.pkl','wb'))

from tensorflow.keras.preprocessing import image


# filenames=pickle.load(open('filenames.pkl','rb'))

model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
print(model.summary())

def feature_extractor(img_path,model):
    img1 = image.load_img(img_path,target_size=(224,224))
    img1_array = image.img_to_array(img1)
    expanded_img1 = np.expand_dims(img1_array,axis=0)
    preprocessed_img1 = preprocess_input(expanded_img1)

    result1 = model.predict(preprocessed_img1).flatten()

    return result1

features=[]

for file in tqdm(filenames):
    features.append(feature_extractor(file,model))


# pickle.dump(features,open('embedding.pkl','wb'))





# feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
# filenames = pickle.load(open('filenames.pkl','rb'))

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

detector = MTCNN()
# load img -> face detection
sample_img = cv2.imread('sample/HRoshan.jpg')
result2 = detector.detect_faces(sample_img)

x,y,width,height = result2[0]['box']

face = sample_img[y:y+height,x:x+width]

cv2.imshow('output',face)
cv2.waitKey(0)

image3=Image.fromarray(face)
image3=Image.resize((224,224))

face_array= np.asarray(image3)
face_array=face_array.astype('float32')

expanded_img3 = np.expand_dims(face_array,axis=0)
preprocessed_img3 = preprocess_input(expanded_img3)

result3 = model.predict(preprocessed_img3).flatten()

# print(result)
# print(result.shape)


# find the cosine distance of current image with all the 8655 features
similarity1 = []
for i in range(len(feature_list)):
    similarity1.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])

index_pos1 = sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]

# temp_img = cv2.imread(filenames[index_pos])
# cv2.imshow('output',temp_img)
# cv2.waitKey(0)


from mtcnn import MTCNN
import numpy as np

detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
# feature_list = pickle.load(open('embedding.pkl','rb'))
# filenames = pickle.load(open('filenames.pkl','rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    #  extract its features
    image3 = Image.fromarray(face)
    image3 = image3.resize((224, 224))

    face_array = np.asarray(image3)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('Which bollywood celebrity are you?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)

        # extract the features
        features = extract_features(os.path.join('uploads',uploaded_image.name),model,detector)
        # recommend
        index_pos = recommend(feature_list,features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        # display
        col1,col2 = st.beta_columns(2)

        with col1:
            st.header('Your uploaded image')
            st.image(display_image)
        with col2:
            st.header("Seems like " + predicted_actor)
            st.image(filenames[index_pos],width=300)
