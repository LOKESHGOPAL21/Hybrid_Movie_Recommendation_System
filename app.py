from flask import Flask, request, jsonify
import requests
from models import content_based_recommend, collaborative_filtering_recommend

app = Flask(__name__)
TMDB_API_KEY = "YOUR_TMDB_API_KEY"

# Fetch movie ID from TMDb API
def get_movie_id(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    response = requests.get(url).json()
    if response['results']:
        return response['results'][0]['id'], response['results'][0]['title']
    return None, None

@app.route('/recommend', methods=['GET'])
def recommend():
    movie_name = request.args.get('movie')
    if not movie_name:
        return jsonify({"error": "Please provide a movie name"}), 400

    movie_id, _ = get_movie_id(movie_name)
    if not movie_id:
        return jsonify({"error": "Movie not found"}), 404

    content_recs = content_based_recommend(movie_name)
    collab_recs = collaborative_filtering_recommend(movie_id)

    final_recommendations = content_recs + collab_recs
    final_recommendations = {v['title']: v for v in final_recommendations}.values()  # Remove duplicates

    return jsonify({"recommendations": list(final_recommendations)})

if __name__ == '__main__':
    app.run(debug=True)
