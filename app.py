from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "cleaned_inventory_data.csv"
MODEL_FILE = BASE_DIR / "demand_model.pkl"


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.dropna(
    subset=["Date"]
)


# ============================================================
# 3. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",
    "Demand"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)


# ============================================================
# 4. PREPARE PRODUCT DATA
# ============================================================

df["Product ID"] = df["Product ID"].astype(str)

df = df.sort_values(
    ["Product ID", "Date"]
).reset_index(drop=True)


# ============================================================
# 5. LOAD TRAINED ML MODEL
# ============================================================

model = joblib.load(MODEL_FILE)


# ============================================================
# 6. MODEL FEATURE ORDER
# ============================================================

FEATURES = [

    "Day",
    "Month",
    "DayOfWeek",
    "Year",

    "Lag_1",
    "Lag_7",
    "Rolling_Mean_7",

    "Inventory Level",
    "Units Ordered",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic"

]


# ============================================================
# 7. DEMAND FORECASTING FUNCTION
# ============================================================

def forecast_future_demand(product_id, days=7):

    product_df = df[
        df["Product ID"] == str(product_id)
    ].copy()

    product_df = product_df.sort_values("Date")


    # Product not found
    if product_df.empty:
        return []


    # Historical demand
    history = (
        product_df["Demand"]
        .astype(float)
        .tolist()
    )


    # Latest inventory information
    latest = product_df.iloc[-1]


    # Last available date
    last_date = product_df["Date"].max()


    predictions = []


    # Generate future predictions
    for day in range(1, days + 1):

        future_date = (
            last_date +
            pd.Timedelta(days=day)
        )


        # -------------------------
        # Lag 1
        # -------------------------

        lag_1 = history[-1]


        # -------------------------
        # Lag 7
        # -------------------------

        if len(history) >= 7:

            lag_7 = history[-7]

        else:

            lag_7 = history[0]


        # -------------------------
        # Rolling Mean 7
        # -------------------------

        recent_values = history[-7:]

        rolling_mean_7 = (
            sum(recent_values) /
            len(recent_values)
        )


        # -------------------------
        # Create model input
        # -------------------------

        input_data = pd.DataFrame(

            [{

                "Day": future_date.day,

                "Month": future_date.month,

                "DayOfWeek": future_date.dayofweek,

                "Year": future_date.year,

                "Lag_1": lag_1,

                "Lag_7": lag_7,

                "Rolling_Mean_7":
                    rolling_mean_7,

                "Inventory Level":
                    latest["Inventory Level"],

                "Units Ordered":
                    latest["Units Ordered"],

                "Price":
                    latest["Price"],

                "Discount":
                    latest["Discount"],

                "Promotion":
                    latest["Promotion"],

                "Competitor Pricing":
                    latest["Competitor Pricing"],

                "Epidemic":
                    latest["Epidemic"]

            }],

            columns=FEATURES
        )


        # -------------------------
        # Predict demand
        # -------------------------

        prediction = float(
            model.predict(input_data)[0]
        )


        # Prevent negative demand
        prediction = max(
            0,
            prediction
        )


        # -------------------------
        # Store prediction
        # -------------------------

        predictions.append({

            "date":
                future_date.strftime(
                    "%Y-%m-%d"
                ),

            "predicted_demand":
                round(
                    prediction,
                    2
                )

        })


        # -------------------------
        # Use prediction for next day
        # -------------------------

        history.append(
            prediction
        )


    return predictions


# ============================================================
# 8. DASHBOARD DATA
# ============================================================

def get_dashboard_data(
    product_id="P0001"
):

    forecast = forecast_future_demand(
        product_id
    )


    total_forecast = None


    if forecast:

        total_forecast = round(

            sum(
                item["predicted_demand"]
                for item in forecast
            ),

            2

        )


    return {

        "total_products":
            int(
                df["Product ID"]
                .nunique()
            ),

        "total_stock":
            int(
                df["Inventory Level"]
                .sum()
            ),

        "total_sales":
            int(
                df["Units Sold"]
                .sum()
            ),

        "total_forecast":
            total_forecast,

        "forecast":
            forecast,

        "selected_product":
            product_id,

        "product_ids":
            sorted(
                df["Product ID"]
                .unique()
                .tolist()
            )

    }


# ============================================================
# 9. MAIN DASHBOARD
# ============================================================

@app.route("/")
def home():

    product_id = request.args.get(
        "product",
        "P0001"
    )


    data = get_dashboard_data(
        product_id
    )


    return render_template(
        "index.html",
        **data
    )


# ============================================================
# 10. DEMAND FORECAST PAGE
# ============================================================

@app.route("/forecast")
def forecast_page():

    product_id = request.args.get(
        "product",
        "P0001"
    )


    data = get_dashboard_data(
        product_id
    )


    return render_template(
        "forecast.html",
        **data
    )


# ============================================================
# 11. DASHBOARD API
# ============================================================

@app.route("/api/dashboard")
def dashboard_api():

    product_id = request.args.get(
        "product",
        "P0001"
    )


    return jsonify(
        get_dashboard_data(
            product_id
        )
    )


# ============================================================
# 12. FORECAST API
# ============================================================

@app.route(
    "/api/forecast/<product_id>"
)
def forecast_api(product_id):

    forecast = forecast_future_demand(
        product_id
    )


    # Product not found
    if not forecast:

        return jsonify({

            "error":
                "Product not found or forecast unavailable"

        }), 404


    # Calculate total forecast
    total = round(

        sum(
            item["predicted_demand"]
            for item in forecast
        ),

        2

    )


    return jsonify({

        "product_id":
            product_id,

        "forecast":
            forecast,

        "total_forecast":
            total

    })


# ============================================================
# 13. RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "Dataset loaded successfully!"
    )

    print(
        "ML model loaded successfully!"
    )

    print(
        "Starting Smart Inventory Intelligence System..."
    )

    app.run(
        debug=True
    )