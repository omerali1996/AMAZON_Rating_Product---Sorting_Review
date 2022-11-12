import pandas as pd
import math
import scipy.stats as st
import datetime as dt

pd.set_option("display.max_row",None)
pd.set_option("display.float_format",lambda x: "%.5f" %x)


data = pd.read_csv("Cases/amazon_review.csv")
df=data.copy()

df.head()
df.shape
df.columns
df.info()

#adım1
df["overall"].mean()

#adım2
df["reviewTime"]=pd.to_datetime(df["reviewTime"])

current_date= df["reviewTime"].max()

df["difference"]=(current_date - df["reviewTime"]).dt.days
q1,q2,q3=df["difference"].quantile(q=[0.25,0.5,0.75])


# 28 + 26 + 24 + 22 yüzdelikler

df[(df["difference"] <= q1)]["overall"].mean() *28/100
df[(df["difference"] <= q1)]["overall"].value_counts()
df[(df["difference"] > q1)&(df["difference"] <= q2)]["overall"]\
    .mean() *26/100
df[(df["difference"] > q1)&(df["difference"] <= q2)]["overall"]\
    .value_counts()
df[(df["difference"] > q2) & (df["difference"] <= q3)]["overall"]\
    .mean()*24/100

df[(df["difference"] > q3)]["overall"].mean()*22/100

df[(df["difference"] > q3)]["overall"].mean()


# adım"1


df["helpful_no"]=df["total_vote"] - df["helpful_yes"]

def score_up_down_diff(up, down):
    return up - down

df["score_pas_neg_diff"]=df.apply(lambda x:score_up_down_diff(
    x["helpful_yes"],
    x["helpful_no"]),axis=1)


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


df["score_average_rating"]=df.apply(lambda x: score_average_rating(
    x["helpful_yes"],
    x["helpful_no"]),axis=1)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"]=df.apply(lambda x: wilson_lower_bound(
    x["helpful_yes"],
    x["helpful_no"]),axis=1)


df.sort_values(by="wilson_lower_bound",ascending=False).head(10)
pd.set_option("display.max_column",None)

df.columns
df.groupby("overall").agg({"score_pas_neg_diff":"mean",
                        "wilson_lower_bound":"mean",
                        "score_average_rating":"mean"})


df.head(20)