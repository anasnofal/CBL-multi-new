from .BaseController import BaseController
import pandas as pd


class SavingsController(BaseController):

    def calculate_crime_savings(
    self,
    crime_costs_df: pd.DataFrame,
    crime_type: str,
    predicted_count: int,
    ):
        crime_data = crime_costs_df.set_index(crime_costs_df["Crime"].str.strip().str.lower()).loc[crime_type.strip().lower()]

        cost = float(crime_data["Cost"])
        lower_q = float(crime_data["Lower quartile"])
        upper_q = float(crime_data["Upper quartile"])

        effectiveness = float(crime_data["Effectiveness"])

        estimated = cost * predicted_count * effectiveness
        lower = lower_q * predicted_count * effectiveness
        upper = upper_q * predicted_count * effectiveness

        return {
            "crime_type": crime_type,
            "predicted_count": predicted_count,
            "cost": round(cost, 0),
            "effectiveness": effectiveness,
            "estimated_savings": round(estimated, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "recommendation_l1": crime_data["Recommendation L1"],
            "description": crime_data["Description"],
            "source": crime_data["Source"],
        }