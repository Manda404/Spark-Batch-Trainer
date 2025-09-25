import streamlit as st
from datetime import datetime

# ==============================
# Page Configuration
# ==============================
def setup_page():
    st.set_page_config(
        page_title="⚡ Spark Batch Trainer - AXA Data Science",
        page_icon="📊",
        layout="wide"
    )
    st.title("⚡ Spark Batch Trainer - AXA Data Science Platform")
    st.markdown("### 🚀 Batch-wise Training Framework for Scalable Machine Learning")


# ==============================
# Sidebar
# ==============================
def display_sidebar():
    st.sidebar.header("📊 Spark Batch Trainer - AXA")

    # Company logo
    st.sidebar.image("app/images/axa_logo.png", use_container_width=True)

    # App version
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; font-size: 16px; font-weight: bold;">
            📅 Version: 1.0.0 <br/>
            🔄 Last update: {datetime.now().strftime("%Y-%m-%d")}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Framework selection
    st.sidebar.markdown("### 🔧 Select ML Framework")
    framework = st.sidebar.radio("Choose:", ["XGBoost", "CatBoost", "LightGBM"])

    # Config toggles
    st.sidebar.markdown("### ⚙️ Configurations")
    config_model = st.sidebar.checkbox("Use custom model config")
    config_training = st.sidebar.checkbox("Use training config")
    config_lr = st.sidebar.checkbox("Enable learning rate scheduler")

    return framework, config_model, config_training, config_lr


# ==============================
# Dashboard metrics
# ==============================
def display_metrics(train_auc, valid_auc, train_loss, valid_loss):
    st.markdown("## 📊 Training Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Train AUC", f"{train_auc:.2f}")
        st.metric("Validation AUC", f"{valid_auc:.2f}")
    with col2:
        st.metric("Train Logloss", f"{train_loss:.2f}")
        st.metric("Validation Logloss", f"{valid_loss:.2f}")


# ==============================
# Learning curve visualization
# ==============================
def display_learning_curve():
    st.markdown("### 📈 Learning Curve")
    # 👉 Ici tu intègres plotly ou matplotlib
    # Exemple: plotly.express.line(...)
    st.line_chart([0.5, 0.4, 0.35, 0.32, 0.30])


# ==============================
# Main app
# ==============================
def main():
    setup_page()
    framework, cfg_model, cfg_training, cfg_lr = display_sidebar()
    
    # Dummy results (à remplacer par les vrais logs Spark Batch Trainer)
    display_metrics(train_auc=0.89, valid_auc=0.85, train_loss=0.35, valid_loss=0.40)
    display_learning_curve()


if __name__ == "__main__":
    main()
