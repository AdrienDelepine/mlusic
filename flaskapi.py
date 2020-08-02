import os
from flask import Flask, request, jsonify
import ml


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'songs.csv'),
    )
    app.config["DEBUG"] = True

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/', methods=['GET'])
    def home():
        return "<h1>Mlusic</h1><p>This site is a prototype API for mlusic.</p>"

    @app.route('/all', methods=['GET'])
    def api_all():
        return jsonify(ml.get_songs().to_json())

    @app.route('/songs', methods=['GET'])
    def api_filter():
        query_parameters = request.args
        artist = query_parameters.get('artist')
        title = query_parameters.get('title')

        columns = []
        rows = []
        if artist:
            columns.append('artists')
            rows.append(artist)
        if title:
            columns.append('title')
            rows.append(title)

        songs = ml.pick_rows(columns, rows)
        return jsonify(songs.to_json())

    @app.route('/lyricsim', methods=['GET'])
    def lyricsim():
        if 'title' in request.args:
            title = request.args['title']
        else:
            return "Error: No id field provided. Please specify an id."
        return ml.most_lyrically_similar(title)

    @app.route('/audiosim', methods=['GET'])
    def audiosim():
        if 'title' in request.args:
            title = request.args['title']
        else:
            return "Error: No id field provided. Please specify an id."
        return ml.get_audio_features_NN()

    return app


app = create_app()
app.run()
