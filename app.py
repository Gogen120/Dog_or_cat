from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from dogs_cats_nn import predict_label


app = Flask(__name__)
app.config['UPLOADS_DEFAULT_DEST'] = 'static'
images = UploadSet('images', IMAGES)
configure_uploads(app, (images,))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def load_image():
    if 'picture' in request.files:
        filename = images.save(request.files['picture'])
        return redirect(url_for('show_image', filename=filename))


@app.route('/<filename>')
def show_image(filename):
    predicted_label = predict_label(filename)
    image_url = images.url(filename)
    return render_template('index.html',
                           img=image_url,
                           predicted_label=predicted_label)


if __name__ == '__main__':
    app.run()
