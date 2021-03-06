from flask import Flask, request
import utils

app = Flask(__name__)

HOTELS_CSV_PATH = "./data/hotels.csv"


@app.route('/candidates')
def candidates():
    if "HotelID" not in list(request.args.keys()):
        app.logger.error("HotelID parameter is missing")
        raise ValueError("HotelID is a mandatory parameter")
    else:
        hotel_id = request.args.get("HotelID")
        limit = request.args.get("limit", 5)
        app.logger.info("Finding candidates for hotel id " + request.args.get("HotelID"))
        result_df = utils.top_n_kdtree(int(hotel_id), int(limit), verbose=False)
        subset = ["recommended_hotel_id", "HotelName", "distance_to_input_hotel"]
        return result_df[subset].to_json(orient="records")


def start():
    print("Loading hotels...")
    utils.load_hotels(HOTELS_CSV_PATH)
    print("Processing hotels...")
    utils.process_hotels()
    print("Building KD-Tree geo index...")
    utils.index_hotels_geo()


if __name__ == '__main__':
    start()
    app.run(host='0.0.0.0',debug=True)
