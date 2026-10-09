import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Course Marketplace Forecast", page_icon="📚", layout="centered")
st.title("📚 Course Marketplace Predictor")
st.caption("Income & expense forecasting and course enrollment prediction (simulated data)")


@st.cache_resource
def load_models():
    inc = pd.read_csv("monthly_income.csv", parse_dates=["Date"]).set_index("Date")
    hw = ExponentialSmoothing(inc["Income"], trend="add", damped_trend=True,
                              seasonal="mul", seasonal_periods=12).fit(optimized=True)
    lr_e = LinearRegression().fit(inc[["Income"]], inc["Expenses"])
    d = pd.read_csv("courses.csv")
    X = pd.get_dummies(d.drop(columns=["course_id", "enrollments"]), columns=["category", "level"])
    y = np.log1p(d["enrollments"])
    lin = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3,
                               random_state=42, n_jobs=-1).fit(X, y)
    return inc, hw, lr_e, d, X, lin, rf


inc, hw, lr_e, d, X, lin, rf = load_models()
tab1, tab2 = st.tabs(["📈 Forecast", "🎓 Course Enrollment"])

with tab1:
    n = st.slider("Months to forecast", 3, 24, 12)
    fc = hw.forecast(n)
    fe = pd.Series(lr_e.predict(fc.to_frame("Income")), index=fc.index)
    out = pd.DataFrame({"Income": fc.round(-2), "Expenses": fe.round(-2)})
    out["Surplus"] = out["Income"] - out["Expenses"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total income", f"₹{out['Income'].sum():,.0f}")
    c2.metric("Total expenses", f"₹{out['Expenses'].sum():,.0f}")
    c3.metric("Surplus", f"₹{out['Surplus'].sum():,.0f}")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(inc.index[-24:], inc["Income"].iloc[-24:] / 1e3, color="#1f4e79", label="Income (history)")
    ax.plot(fc.index, fc / 1e3, "--", color="#1f4e79", label="Forecast income")
    ax.plot(fc.index, fe / 1e3, "-.", color="#d9534f", label="Forecast expenses")
    ax.set_ylabel("₹ thousand"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    st.pyplot(fig)
    o = out.copy(); o.index = o.index.strftime("%b %Y")
    st.dataframe(o.style.format("₹{:,.0f}"), width="stretch")
    st.download_button("Download forecast CSV", out.to_csv().encode(), "forecast.csv", "text/csv")

with tab2:
    st.write("Enter course details to predict enrollments.")
    cat = st.selectbox("Category", sorted(d["category"].unique()))
    lvl = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
    price = st.select_slider("Price (₹)", [0, 299, 499, 799, 999, 1499, 1999, 2999, 3999], value=799)
    disc = st.select_slider("Discount (%)", [0, 10, 20, 30, 40, 50], value=0)
    rating = st.slider("Rating", 3.0, 5.0, 4.3, 0.1)
    hours = st.slider("Duration (hours)", 1.0, 40.0, 12.0, 0.5)
    lectures = st.slider("Number of lectures", 5, 200, 50)
    exp = st.slider("Instructor experience (years)", 1, 20, 5)
    age = st.slider("Course age (months)", 1, 48, 12)
    cert = st.checkbox("Certificate included", True)
    model_name = st.radio("Model", ["Linear Regression", "Random Forest"], horizontal=True)
    row = pd.DataFrame([{"price_inr": price, "duration_hours": hours, "num_lectures": lectures,
                         "instructor_experience_yrs": exp, "rating": rating, "discount_pct": disc,
                         "has_certificate": int(cert), "course_age_months": age,
                         "category": cat, "level": lvl}])
    row = pd.get_dummies(row, columns=["category", "level"]).reindex(columns=X.columns, fill_value=0)
    m = lin if model_name == "Linear Regression" else rf
    pred = float(np.expm1(m.predict(row)[0]))
    st.metric("Predicted enrollments", f"{max(pred, 0):,.0f}")
    st.caption("Estimated revenue: ₹{:,.0f}".format(max(pred, 0) * price * (1 - disc / 100)))
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(6)
    st.bar_chart(imp)
