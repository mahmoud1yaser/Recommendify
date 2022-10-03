from flask import render_template, request, Flask
import model as m

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend_fast')
def recommend_fast():
    return render_template('fast.html')


@app.route('/recommend_accurate')
def recommend_accurate():
    return render_template('accurate.html')


@app.route('/show_fast_recommend', methods=['GET', 'POST'])
def show_fast_recommend():
    formValues = [j for j in request.form.values()]
    name = str(formValues[0])
    number = int(formValues[1])
    if m.track_exist_fast(name):
        names_id, artist_names_id, album_names_id, names, artist_names, album_names = m.recommend_me_by_cluster(name,
                                                                                                                number)
        count_songs = len(names)
        return render_template('result.html', names=names, artist_names=artist_names, album_names=album_names,
                               count_songs=count_songs, names_id=names_id, artist_names_id=artist_names_id,
                               album_names_id=album_names_id)
    else:
        not_found = '{} not found in songs library.'.format(name)
        return render_template('result.html', not_found=not_found)


@app.route('/show_accurate_recommend', methods=['GET', 'POST'])
def show_accurate_recommend():
    formValues = [j for j in request.form.values()]
    name = str(formValues[0])
    number = int(formValues[1])
    if m.track_exist_accurate(name):
        names_id, artist_names_id, album_names_id, names, artist_names, album_names = m.recommend_me_by_content(name,
                                                                                                                number)
        count_songs = len(names)
        return render_template('result.html', names=names, artist_names=artist_names, album_names=album_names,
                               count_songs=count_songs, names_id=names_id, artist_names_id=artist_names_id,
                               album_names_id=album_names_id)
    else:
        not_found = '{} not found in songs library.'.format(name)
        return render_template('result.html', not_found=not_found)


if __name__ == '__main__':
    app.run(debug=True)
