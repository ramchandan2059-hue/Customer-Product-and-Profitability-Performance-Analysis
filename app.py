"""
Customer, Product & Profitability Performance Dashboard
APL Logistics (KWE Group) | Unified Mentor project

Run locally:   streamlit run app.py
Data lookup order (first one found is used; searched in the app folder, ./data and the working folder):
  1. clean_data.parquet   (compact file, recommended for deployment)
  2. APL_Logistics.csv    (raw file, cleaned inside the app)
  3. a file uploaded through the sidebar
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

st.set_page_config(page_title="APL Logistics | Profitability Analytics", page_icon="📦", layout="wide")

BASE = Path(__file__).parent
SEARCH_DIRS = [BASE, BASE / "data", BASE / "data" / "raw", Path.cwd(), Path.cwd() / "data"]


def find_file(patterns):
    """Return the first file matching any pattern (case-insensitive) in the search folders."""
    for folder in SEARCH_DIRS:
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.is_file() and any(p.match(pat) or p.name.lower() == pat.lower() for pat in patterns):
                return p
    return None


PARQUET = find_file(["clean_data.parquet"])
RAW_CSV = find_file(["APL_Logistics.csv", "apl_logistics*.csv"])
EXCLUDE_STATUS = ["CANCELED", "SUSPECTED_FRAUD"]
NEEDED = [
    "Customer Id", "Customer Segment", "Category Name", "Product Name", "Market", "Order Region",
    "Order Country", "Order Status", "Shipping Mode", "Delivery Status", "Type", "Department Name",
    "Order Item Discount", "Order Item Discount Rate", "Order Item Quantity", "Order Item Product Price",
    "Order Item Profit Ratio", "Sales", "Order Item Total", "Order Profit Per Order",
]
BAND_LABELS = ["0%", "1-5%", "6-10%", "11-15%", "16-20%", "20%+"]
TIER_LABELS = ["Low", "Medium", "High", "Platinum"]


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading data...")
def read_source(uploaded_bytes=None):
    if uploaded_bytes is not None:
        import io
        try:
            raw = pd.read_csv(io.BytesIO(uploaded_bytes), encoding="latin-1")
        except Exception:
            raw = pd.read_csv(io.BytesIO(uploaded_bytes))
    elif PARQUET is not None:
        raw = pd.read_parquet(PARQUET)
    elif RAW_CSV is not None:
        raw = pd.read_csv(RAW_CSV, encoding="latin-1", usecols=NEEDED)
    else:
        return None
    return raw


@st.cache_data(show_spinner="Preparing data...")
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[NEEDED].copy()
    df = df[~df["Order Status"].isin(EXCLUDE_STATUS)]
    df = df[(df["Sales"] > 0) & (df["Order Item Quantity"] > 0) & (df["Order Item Total"] > 0)]
    df["Order Country"] = df["Order Country"].replace({"EE. UU.": "USA"})
    df["profit"] = df["Order Profit Per Order"]
    df["net_revenue"] = df["Order Item Total"]
    df["margin_pct"] = df["profit"] / df["net_revenue"] * 100
    df["discount_pct"] = df["Order Item Discount Rate"] * 100
    df["is_loss"] = (df["profit"] < 0).astype(int)
    df["has_discount"] = (df["Order Item Discount"] > 0).astype(int)
    bins = [-0.001, 0, 5, 10, 15, 20, 100]
    df["discount_band"] = pd.cut(df["discount_pct"].round(2), bins=bins, labels=BAND_LABELS)
    for c in ["Customer Segment", "Category Name", "Product Name", "Market", "Order Region",
              "Order Country", "Order Status", "Shipping Mode", "Delivery Status", "Type", "Department Name"]:
        df[c] = df[c].astype("category")
    return df.reset_index(drop=True)


def summarise(frame: pd.DataFrame, by) -> pd.DataFrame:
    g = frame.groupby(by, observed=True).agg(
        revenue=("net_revenue", "sum"),
        sales=("Sales", "sum"),
        profit=("profit", "sum"),
        orders=("profit", "size"),
        loss_orders=("is_loss", "sum"),
        avg_discount=("discount_pct", "mean"),
    )
    g["margin_pct"] = g["profit"] / g["revenue"] * 100
    g["loss_share_pct"] = g["loss_orders"] / g["orders"] * 100
    return g


def fmt_money(x):
    ax = abs(x)
    if ax >= 1e6:
        return f"{x/1e6:,.2f}M"
    if ax >= 1e3:
        return f"{x/1e3:,.1f}K"
    return f"{x:,.0f}"


# --------------------------------------------------------------------------- load
uploaded = None
raw = read_source()
if raw is None:
    st.sidebar.warning("No data file found next to app.py.")
    st.sidebar.caption("Looked in: " + ", ".join(str(d) for d in SEARCH_DIRS if d.is_dir()))
    up = st.sidebar.file_uploader("Upload APL_Logistics.csv", type="csv")
    if up is None:
        st.title("📦 APL Logistics | Profitability Analytics")
        st.info("Add `clean_data.parquet` or `APL_Logistics.csv` next to `app.py`, or upload the CSV in the sidebar.")
        st.stop()
    raw = read_source(up.getvalue())

missing = [c for c in NEEDED if c not in raw.columns]
if missing:
    st.error(f"The data is missing required columns: {missing}")
    st.stop()

df = prepare(raw)

# --------------------------------------------------------------------------- sidebar
st.sidebar.title("📦 APL Logistics")
page = st.sidebar.radio(
    "Dashboard",
    ["Revenue & Profit Overview", "Customer Value", "Product & Category", "Discount Impact Analyzer", "Market & Region"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

segments = sorted(df["Customer Segment"].unique())
sel_seg = st.sidebar.multiselect("Customer segment", segments, default=segments)

markets = ["All"] + sorted(df["Market"].unique())
sel_market = st.sidebar.selectbox("Market", markets)

region_pool = df if sel_market == "All" else df[df["Market"] == sel_market]
regions = ["All"] + sorted(region_pool["Order Region"].unique())
sel_region = st.sidebar.selectbox("Order region", regions)

cats = sorted(df["Category Name"].unique())
sel_cat = st.sidebar.multiselect("Category (empty = all)", cats)

prod_pool = df if not sel_cat else df[df["Category Name"].isin(sel_cat)]
prods = sorted(prod_pool["Product Name"].unique())
sel_prod = st.sidebar.multiselect("Product (empty = all)", prods)

dmax = float(np.ceil(df["discount_pct"].max()))
sel_disc = st.sidebar.slider("Discount rate (%)", 0.0, dmax, (0.0, dmax), step=1.0)

f = df[df["Customer Segment"].isin(sel_seg)]
if sel_market != "All":
    f = f[f["Market"] == sel_market]
if sel_region != "All":
    f = f[f["Order Region"] == sel_region]
if sel_cat:
    f = f[f["Category Name"].isin(sel_cat)]
if sel_prod:
    f = f[f["Product Name"].isin(sel_prod)]
f = f[f["discount_pct"].between(sel_disc[0], sel_disc[1])]

st.sidebar.caption(f"{len(f):,} of {len(df):,} order lines selected")
st.sidebar.caption(
    "Canceled and suspected-fraud orders are excluded. "
    "Margin = profit / order item total (net of discount)."
)

st.title("📦 Customer, Product & Profitability Performance")
if f.empty:
    st.warning("No orders match the current filters. Widen the filters in the sidebar.")
    st.stop()

tot_rev, tot_net, tot_profit = f["Sales"].sum(), f["net_revenue"].sum(), f["profit"].sum()
tot_margin = tot_profit / tot_net * 100


def kpi_row():
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total revenue (sales)", fmt_money(tot_rev))
    c2.metric("Total profit", fmt_money(tot_profit))
    c3.metric("Profit margin", f"{tot_margin:.2f}%")
    c4.metric("Loss-making orders", f"{f['is_loss'].mean()*100:.1f}%")
    c5.metric("Avg discount", f"{f['discount_pct'].mean():.1f}%")


# =========================================================================== PAGE 1
if page == "Revenue & Profit Overview":
    st.header("Revenue & Profit Overview")
    kpi_row()
    st.caption("The dataset has no order date, so comparisons below use order status, shipping mode and discount band instead of a time trend.")

    c1, c2 = st.columns(2)
    dim = c1.selectbox("Compare margin by", ["Shipping Mode", "Order Status", "Delivery Status", "Type", "Department Name", "Customer Segment"])
    g = summarise(f, dim).reset_index().sort_values("revenue", ascending=False)
    fig = px.bar(g, x=dim, y=["revenue", "profit"], barmode="group", title=f"Revenue vs profit by {dim}")
    c1.plotly_chart(fig)
    fig = px.line(g, x=dim, y="margin_pct", markers=True, title=f"Margin % by {dim}")
    fig.update_yaxes(rangemode="tozero")
    c2.plotly_chart(fig)

    st.subheader("Profit concentration (Pareto)")
    by = st.radio("Rank by", ["Customer Id", "Product Name"], horizontal=True)
    p = f.groupby(by, observed=True)["profit"].sum().sort_values(ascending=False)
    pos = p[p > 0]
    cum = pos.cumsum() / pos.sum() * 100
    n80 = int((cum <= 80).sum() + 1)
    st.write(f"**{n80:,} of {len(p):,} ({n80/len(p):.1%})** generate 80% of positive profit.")
    pareto = pd.DataFrame({"pct_of_entities": np.arange(1, len(cum) + 1) / len(p) * 100, "cum_profit_pct": cum.values})
    fig = px.line(pareto, x="pct_of_entities", y="cum_profit_pct", title="Cumulative profit share")
    fig.add_hline(y=80, line_dash="dash", line_color="red")
    fig.update_layout(xaxis_title="% of entities (ranked by profit)", yaxis_title="Cumulative % of positive profit")
    st.plotly_chart(fig)

    st.subheader("Margin distribution per order")
    fig = px.histogram(f, x=f["margin_pct"].clip(-100, 60), nbins=60, labels={"x": "Margin % (clipped to -100..60 for display)"})
    st.plotly_chart(fig)

# =========================================================================== PAGE 2
elif page == "Customer Value":
    st.header("Customer Value Dashboard")
    cust = f.groupby("Customer Id", observed=True).agg(
        segment=("Customer Segment", "first"),
        revenue=("net_revenue", "sum"),
        profit=("profit", "sum"),
        orders=("profit", "size"),
        avg_discount=("discount_pct", "mean"),
    )
    cust["margin_pct"] = cust["profit"] / cust["revenue"] * 100
    avg_profit = cust["profit"].mean()
    cust["customer_value_index"] = cust["profit"] / avg_profit if avg_profit else np.nan
    if len(cust) >= 4:
        cust["tier"] = pd.qcut(cust["profit"].rank(method="first"), 4, labels=TIER_LABELS)
    else:
        cust["tier"] = "Low"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(cust):,}")
    c2.metric("Avg profit / customer", f"{avg_profit:,.0f}")
    c3.metric("Loss-making customers", f"{(cust.profit < 0).sum():,} ({(cust.profit < 0).mean()*100:.1f}%)")
    c4.metric("Top customer value index", f"{cust.customer_value_index.max():.1f}x")

    n = st.slider("Customers to show", 5, 25, 10)
    t1, t2 = st.columns(2)
    top = cust.sort_values("profit", ascending=False).head(n).reset_index()
    bot = cust.sort_values("profit").head(n).reset_index()
    top["Customer Id"] = top["Customer Id"].astype(str)
    bot["Customer Id"] = bot["Customer Id"].astype(str)
    t1.plotly_chart(px.bar(top, x="profit", y="Customer Id", orientation="h", color="segment", title=f"Top {n} customers by profit").update_yaxes(autorange="reversed"))
    t2.plotly_chart(px.bar(bot, x="profit", y="Customer Id", orientation="h", color="segment", title=f"Bottom {n} customers by profit").update_yaxes(autorange="reversed"))

    st.subheader("Customer value tiers (profit quartiles)")
    tier = cust.groupby("tier", observed=True).agg(
        customers=("profit", "size"), revenue=("revenue", "sum"), profit=("profit", "sum"),
        avg_margin_pct=("margin_pct", "mean"), avg_discount=("avg_discount", "mean"))
    tier["profit_share_pct"] = tier["profit"] / tier["profit"].sum() * 100
    tier["revenue_share_pct"] = tier["revenue"] / tier["revenue"].sum() * 100
    a, b = st.columns([2, 1])
    a.dataframe(tier.round(2))
    b.plotly_chart(px.bar(tier.reset_index(), x="tier", y="profit", title="Profit by tier"))

    st.subheader("Segment contribution")
    seg = summarise(f, "Customer Segment")
    seg["customers"] = f.groupby("Customer Segment", observed=True)["Customer Id"].nunique()
    seg["profit_per_customer"] = seg["profit"] / seg["customers"]
    s1, s2, s3 = st.columns(3)
    s1.plotly_chart(px.pie(seg.reset_index(), names="Customer Segment", values="profit", title="Profit share"))
    s2.plotly_chart(px.bar(seg.reset_index(), x="Customer Segment", y="margin_pct", title="Margin %"))
    s3.plotly_chart(px.bar(seg.reset_index(), x="Customer Segment", y="profit_per_customer", title="Profit per customer"))

    st.subheader("High revenue, weak profit customers")
    risky = cust[(cust.revenue >= cust.revenue.quantile(0.75)) & (cust.margin_pct <= cust.margin_pct.quantile(0.25))]
    st.write(f"**{len(risky):,}** customers are in the top revenue quartile but the bottom margin quartile.")
    st.dataframe(risky.sort_values("revenue", ascending=False).head(25).round(2))

    st.download_button("Download customer table (CSV)", cust.reset_index().to_csv(index=False).encode(), "customer_summary.csv", "text/csv")

# =========================================================================== PAGE 3
elif page == "Product & Category":
    st.header("Product & Category Performance")
    kpi_row()

    cat = summarise(f, "Category Name").sort_values("profit", ascending=False)
    prod = summarise(f, "Product Name").sort_values("profit", ascending=False)
    loss_cats = cat[cat.profit < 0]
    st.write(f"Loss-making categories: **{len(loss_cats)}** | Loss-making products: **{(prod.profit < 0).sum()}**")

    c1, c2 = st.columns(2)
    k = st.slider("Categories to show", 5, 25, 10)
    c1.plotly_chart(px.bar(cat.head(k).reset_index(), x="profit", y="Category Name", orientation="h", title=f"Top {k} categories by profit").update_yaxes(autorange="reversed"))
    c2.plotly_chart(px.bar(cat.sort_values("margin_pct").head(k).reset_index(), x="margin_pct", y="Category Name", orientation="h", title=f"Lowest {k} category margins (%)").update_yaxes(autorange="reversed"))

    st.subheader("Product revenue vs margin")
    if len(prod) >= 4:
        rev_cut, mar_cut = prod.revenue.quantile(0.75), prod.margin_pct.median()
        prod["flag"] = np.where((prod.revenue >= rev_cut) & (prod.margin_pct < mar_cut), "High revenue, low margin", "Other")
    else:
        prod["flag"] = "Other"
    fig = px.scatter(prod.reset_index(), x="revenue", y="margin_pct", color="flag", size="orders", hover_name="Product Name",
                     color_discrete_map={"High revenue, low margin": "crimson", "Other": "steelblue"})
    if len(prod) >= 4:
        fig.add_hline(y=mar_cut, line_dash="dash", line_color="grey")
        fig.add_vline(x=rev_cut, line_dash="dash", line_color="grey")
    st.plotly_chart(fig)
    st.dataframe(prod[prod.flag == "High revenue, low margin"].sort_values("margin_pct")[["revenue", "profit", "margin_pct", "avg_discount", "loss_share_pct"]].round(2))

    st.subheader("Category profitability heatmap")
    metric = st.radio("Heatmap metric", ["Margin %", "Profit"], horizontal=True)
    col_dim = st.selectbox("Columns", ["Market", "Order Region", "Customer Segment", "Shipping Mode"])
    pp = f.pivot_table(index="Category Name", columns=col_dim, values="profit", aggfunc="sum", observed=True)
    pr = f.pivot_table(index="Category Name", columns=col_dim, values="net_revenue", aggfunc="sum", observed=True)
    heat = (pp / pr * 100) if metric == "Margin %" else pp
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="RdYlGn", text_auto=".1f",
                    color_continuous_midpoint=float(np.nanmedian(heat.values)), labels={"color": metric})
    fig.update_layout(height=max(400, 24 * len(heat)))
    st.plotly_chart(fig)

    st.subheader("Product table")
    st.dataframe(prod.drop(columns="flag").round(2))
    st.download_button("Download product table (CSV)", prod.reset_index().to_csv(index=False).encode(), "product_summary.csv", "text/csv")

# =========================================================================== PAGE 4
elif page == "Discount Impact Analyzer":
    st.header("Discount Impact Analyzer")
    kpi_row()

    wd = f.groupby("has_discount").agg(orders=("profit", "size"), revenue=("net_revenue", "sum"), profit=("profit", "sum"), loss_share=("is_loss", "mean"))
    wd["margin_pct"] = wd["profit"] / wd["revenue"] * 100
    wd.index = ["No discount" if i == 0 else "Discounted" for i in wd.index]

    band = summarise(f, "discount_band")
    band = band[band.orders > 0]
    m0 = wd.loc["No discount", "margin_pct"] if "No discount" in wd.index else np.nan
    m1 = wd.loc["Discounted", "margin_pct"] if "Discounted" in wd.index else np.nan
    dir_ = (m0 - m1) / m0 if pd.notna(m0) and pd.notna(m1) and m0 != 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Margin, no discount", "n/a" if pd.isna(m0) else f"{m0:.2f}%")
    c2.metric("Margin, discounted", "n/a" if pd.isna(m1) else f"{m1:.2f}%")
    c3.metric("Discount Impact Ratio", "n/a" if pd.isna(dir_) else f"{dir_*100:.1f}%", help="(margin at 0% discount - margin on discounted orders) / margin at 0% discount")

    a, b = st.columns(2)
    a.plotly_chart(px.bar(band.reset_index(), x="discount_band", y="margin_pct", title="Margin % by discount band"))
    b.plotly_chart(px.bar(band.reset_index(), x="discount_band", y="loss_share_pct", title="% loss-making orders by band"))

    rate = f.assign(rate=f["discount_pct"].round()).groupby("rate").agg(profit=("profit", "sum"), revenue=("net_revenue", "sum"), orders=("profit", "size"))
    rate["margin_pct"] = rate["profit"] / rate["revenue"] * 100
    fig = px.line(rate.reset_index(), x="rate", y="margin_pct", markers=True, title="Margin % by exact discount rate")
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig)

    st.subheader("Statistical tests")
    sample = f if len(f) <= 100_000 else f.sample(100_000, random_state=1)
    if sample["Order Item Discount Rate"].nunique() > 1:
        rho, p_rho = stats.spearmanr(sample["Order Item Discount Rate"], sample["Order Item Profit Ratio"])
        st.write(f"Spearman correlation (discount rate vs profit ratio): **rho = {rho:.4f}**, p = {p_rho:.3g}")
    a_ = f.loc[f.has_discount == 0, "Order Item Profit Ratio"]
    b_ = f.loc[f.has_discount == 1, "Order Item Profit Ratio"]
    if len(a_) > 1 and len(b_) > 1:
        t, p_t = stats.ttest_ind(a_, b_, equal_var=False)
        st.write(f"Welch t-test (no discount vs discounted): t = {t:.3f}, p = {p_t:.3g}")
    st.info("A p-value above 0.05 means the data shows no reliable link between discount level and margin. "
            "Report that as a finding rather than assuming discounts always erode margin.")

    margin_floor = st.slider("Margin floor for threshold check (%)", 0.0, 25.0, 10.0, step=0.5)
    below = band[band.margin_pct < margin_floor]
    st.write(f"First discount band below {margin_floor:.1f}% margin: **{below.index[0] if len(below) else 'none'}**")

    st.subheader("What-if: cap the discount rate")
    w1, w2 = st.columns(2)
    cap = w1.slider("Maximum discount allowed (%)", 0, int(dmax), min(15, int(dmax)))
    vol_loss = w2.slider("Volume lost per 1-point discount cut (%)", 0.0, 5.0, 0.0, step=0.5,
                         help="0 = customers buy the same volume. Higher values assume demand falls when discounts are reduced.")
    capped_rate = np.minimum(f["Order Item Discount Rate"], cap / 100)
    saved = f["Order Item Discount"] - capped_rate * f["Sales"]
    cut_pts = (f["Order Item Discount Rate"] - capped_rate) * 100
    vol_factor = np.clip(1 - vol_loss / 100 * cut_pts, 0, 1)
    new_profit = ((f["profit"] + saved) * vol_factor).sum()
    new_net = ((f["net_revenue"] + saved) * vol_factor).sum()
    new_margin = new_profit / new_net * 100 if new_net else np.nan
    affected = (cut_pts > 0).mean() * 100

    m1_, m2_, m3_, m4_ = st.columns(4)
    m1_.metric("Profit now", fmt_money(tot_profit))
    m2_.metric("Profit under cap", fmt_money(new_profit), f"{(new_profit/tot_profit-1)*100:+.1f}%" if tot_profit else None)
    m3_.metric("Margin under cap", f"{new_margin:.2f}%", f"{new_margin - tot_margin:+.2f} pts")
    m4_.metric("Order lines affected", f"{affected:.1f}%")

    caps = list(range(0, int(dmax) + 1, 5))
    rows = []
    for cp in caps:
        cr = np.minimum(f["Order Item Discount Rate"], cp / 100)
        sv = f["Order Item Discount"] - cr * f["Sales"]
        vf = np.clip(1 - vol_loss / 100 * (f["Order Item Discount Rate"] - cr) * 100, 0, 1)
        rows.append({"cap_pct": cp, "profit": ((f["profit"] + sv) * vf).sum()})
    sc = pd.DataFrame(rows)
    fig = px.line(sc, x="cap_pct", y="profit", markers=True, title="Profit by discount cap (current volume assumption)")
    fig.add_hline(y=tot_profit, line_dash="dash", line_color="grey", annotation_text="current profit")
    st.plotly_chart(fig)
    st.caption("Assumes every dollar of discount removed becomes profit, scaled by the volume-loss setting above. "
               "This is a scenario, not a forecast. With 0% volume loss it is an upper bound on the gain.")

# =========================================================================== PAGE 5
else:
    st.header("Market & Regional Profit Analysis")
    kpi_row()

    level = st.radio("Level", ["Market", "Order Region", "Order Country"], horizontal=True)
    g = summarise(f, level)
    if level == "Order Country":
        min_orders = st.slider("Minimum orders per country", 1, 1000, 100)
        g = g[g.orders >= min_orders]
    g["revenue_share_pct"] = g["revenue"] / g["revenue"].sum() * 100
    g["profit_share_pct"] = g["profit"] / g["profit"].sum() * 100
    gg = g.reset_index()

    a, b = st.columns(2)
    a.plotly_chart(px.bar(gg.sort_values("profit", ascending=False).head(20), x=level, y=["revenue", "profit"], barmode="group", title=f"Revenue vs profit by {level}"))
    b.plotly_chart(px.bar(gg.sort_values("margin_pct", ascending=False).head(20), x=level, y="margin_pct", title=f"Margin % by {level}"))

    fig = px.scatter(gg, x="revenue", y="margin_pct", size="orders", hover_name=level, text=level if len(gg) <= 25 else None,
                     title="Revenue vs margin (bubble size = orders)")
    if len(gg) >= 2:
        fig.add_hline(y=gg.margin_pct.median(), line_dash="dash", line_color="grey")
        fig.add_vline(x=gg.revenue.median(), line_dash="dash", line_color="grey")
    st.plotly_chart(fig)
    st.caption("Top-right: strong revenue and margin. Bottom-right: strong revenue but weak profit, the markets to investigate first.")

    st.dataframe(g.sort_values("profit", ascending=False).round(2))
    st.download_button("Download table (CSV)", g.reset_index().to_csv(index=False).encode(), f"{level.lower().replace(' ', '_')}_summary.csv", "text/csv")
