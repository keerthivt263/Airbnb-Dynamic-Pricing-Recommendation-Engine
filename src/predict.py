{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "cb73be58-348c-44fb-a5f5-41a2205ff1bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Model loaded successfully.\n",
      "Loaded feature dataset with shape: (73367, 14)\n",
      "Predicted price: 159.79278869135817\n",
      "✅ Recommended price: 159.79\n"
     ]
    }
   ],
   "source": [
    "# src/predict.py\n",
    "import joblib\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "# Paths\n",
    "MODEL_PATH = os.path.join(\"C:/Users/DELL/Downloads/project/Elevate labs/airbnb_pricing/models/model_lightgbm.joblib\")\n",
    "FEATURES_PATH = os.path.join(\"C:/Users/DELL/Downloads/project/Elevate labs/airbnb_pricing/data/airbnb_features.csv\")\n",
    "\n",
    "def recommend(listing_row: pd.Series, model):\n",
    "    \"\"\"\n",
    "    Takes one listing (row of features) and predicts a recommended price.\n",
    "    Applies simple business rules for safety and realism.\n",
    "    \"\"\"\n",
    "    X = listing_row.to_frame().T\n",
    "    pred_price = model.predict(X)[0]\n",
    "\n",
    "    # Simple rules for final recommended price\n",
    "    floor = max(10, 0.6 * pred_price)     # don't go below 60% of predicted\n",
    "    ceiling = 1.6 * pred_price            # don't go above 160% of predicted\n",
    "    recommended = round(float(np.clip(pred_price, floor, ceiling)), 2)\n",
    "\n",
    "    return recommended\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    # Load model\n",
    "    model = joblib.load(MODEL_PATH)\n",
    "    print(\"✅ Model loaded successfully.\")\n",
    "\n",
    "    # Load sample feature data\n",
    "    df = pd.read_csv(FEATURES_PATH)\n",
    "    print(\"Loaded feature dataset with shape:\", df.shape)\n",
    "\n",
    "    # Select one sample listing (first row)\n",
    "    row = df.drop(columns=[\"price\"]).iloc[0]\n",
    "    rec_price = recommend(row, model)\n",
    "\n",
    "    print(\"Predicted price:\", model.predict(row.to_frame().T)[0])\n",
    "    print(\"✅ Recommended price:\", rec_price)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53b0e01a-035e-4b26-91d0-227a5671cf4d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (airbnb_env)",
   "language": "python",
   "name": "airbnb_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.23"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
