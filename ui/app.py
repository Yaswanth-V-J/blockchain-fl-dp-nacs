import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from web3 import Web3

from ui import train_runner

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Privacy-Preserving Medical FL Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .role-badge-admin {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #93C5FD;
    }
    .role-badge-hospital {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #FDE68A;
    }
    .rank-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 0.6rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .rank-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
    }
    .rank-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .metric-box {
        background-color: #F8FAFC;
        padding: 0.9rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .preview-box {
        background-color: #EFF6FF;
        border: 1px dashed #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def check_blockchain_connection(rpc_url: str = "http://127.0.0.1:8545") -> bool:
    """Check Hardhat Ethereum Node Connection."""
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        return w3.is_connected()
    except Exception:
        return False


def get_pipeline_stage() -> str:
    """Determine Current Pipeline Stage based on completed results."""
    if os.path.exists("results/phase6_nih_multiseed.json"):
        return "Phase 6 - NIH Multi-Seed Cross-Domain Complete"
    elif os.path.exists("results/phase6_nih.json"):
        return "Phase 6 - NIH Multi-Label FL Complete"
    elif os.path.exists("results/phase5_har_multiseed.json"):
        return "Phase 5 - UCI HAR Cross-Domain Complete"
    elif os.path.exists("results/phase4_nacs_multiseed.json"):
        return "Phase 4 - Multi-Seed NACS Robustness Complete"
    elif os.path.exists("results/phase4_nacs.json"):
        return "Phase 4 - NACS Single-Seed Complete"
    elif os.path.exists("results/phase3_connected.json"):
        return "Phase 3 - Blockchain Connected FL Complete"
    elif os.path.exists("results/phase2_dp_eps8.0.json"):
        return "Phase 2 - Differential Privacy Complete"
    elif os.path.exists("results/phase1_baseline.json"):
        return "Phase 1 - Baseline FL Complete"
    return "Initial - Ready to Train"


def load_nacs_data():
    """Load latest NACS experiment data (prefer phase4_nacs.json or phase6_nih.json)."""
    for p in ["results/phase4_nacs.json", "results/phase6_nih.json"]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f), p
            except Exception:
                pass
    return None, None


# -----------------------------------------------------------------------------
# SIDEBAR: VIEW AS ROLE SELECTOR & TRAIN CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Platform Controls")
st.sidebar.markdown("---")

view_role = st.sidebar.selectbox(
    "👤 View As Role:",
    [
        "Admin (Aggregator)",
        "Hospital 1",
        "Hospital 2",
        "Hospital 3",
        "Hospital 4",
        "Hospital 5",
    ],
    index=0,
    help="Select Admin to view full system-wide analytics, or select a Hospital to view its private participant workspace.",
)

is_admin = view_role == "Admin (Aggregator)"
selected_hospital_id = None if is_admin else int(view_role.split(" ")[1]) - 1

st.sidebar.markdown("---")

if is_admin:
    st.sidebar.markdown("<div class='role-badge-admin'>🔑 Active Role: Admin (Aggregator)</div>", unsafe_allow_html=True)
    st.sidebar.caption("Full system visibility across all participating clients, global metrics, and blockchain logs.")
else:
    st.sidebar.markdown(f"<div class='role-badge-hospital'>🏥 Active Role: Hospital {selected_hospital_id + 1}</div>", unsafe_allow_html=True)
    st.sidebar.caption("Participant workspace. Competitor scores and private parameters are masked.")

st.sidebar.markdown("---")


# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🏥 Privacy-Preserving Medical Federated Learning Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Integrated Differential Privacy (DP-SGD), Blockchain Contribution Ledger & Noise-Aware Adaptive Contribution Scoring (NACS)</div>',
    unsafe_allow_html=True,
)

stage = get_pipeline_stage()
node_connected = check_blockchain_connection()

col_h1, col_h2, col_h3 = st.columns([2, 2, 2])
with col_h1:
    st.markdown(f"**Pipeline Stage:** `{stage}`")
with col_h2:
    status_str = "🟢 Connected (127.0.0.1:8545)" if node_connected else "🔴 Disconnected"
    st.markdown(f"**Hardhat Node Status:** {status_str}")
with col_h3:
    st.markdown("**Smart Contract:** `ContributionLedger` (Ethereum)")

st.divider()


# Load NACS data for calculations
nacs_json_data, nacs_source_file = load_nacs_data()

# Compute overall leaderboard ranks if data exists
hospital_ranks = {}
hospital_latest_scores = {}
hospital_epsilons = [1.0, 3.0, 8.0, 15.0, 20.0]
hospital_sizes = [1004, 1782, 203, 424, 1295]  # Standard defaults if file not present

if nacs_json_data and "per_client" in nacs_json_data:
    per_client = nacs_json_data["per_client"]
    for cid in range(5):
        c_info = per_client.get(str(cid), {})
        c_hist = c_info.get("history", [])
        score = c_hist[-1]["nacs_score"] if c_hist else 50
        eps = c_info.get("target_epsilon", hospital_epsilons[cid])
        size = c_info.get("client_size", hospital_sizes[cid])
        hospital_latest_scores[cid] = score
        hospital_epsilons[cid] = eps
        hospital_sizes[cid] = size

    # Rank high to low (Rank 1 = top score)
    sorted_cids = sorted(range(5), key=lambda x: hospital_latest_scores.get(x, 0), reverse=True)
    for rank_idx, cid in enumerate(sorted_cids):
        hospital_ranks[cid] = rank_idx + 1
else:
    for cid in range(5):
        hospital_latest_scores[cid] = 50
        hospital_ranks[cid] = cid + 1


# =============================================================================
# RENDER "TRY THE MODEL" INFERENCE TAB
# =============================================================================
def render_try_the_model_tab():
    st.subheader("🎯 Live Model Inference & Prediction Demo")
    st.info(
        "ℹ️ Research prototype — not a diagnostic tool. Predictions on images outside the training distribution (e.g. non-chest-X-ray uploads) are not meaningful."
    )
    st.markdown("Run a trained global model on real dataset examples or custom user uploads.")

    domain_choice = st.selectbox(
        "Choose Domain / Model:",
        [
            "PneumoniaMNIST (Medical 2D Image)",
            "UCI HAR (Wearable Sensor Data)",
            "NIH Chest X-ray (Multi-Label Clinical Image)",
        ],
        index=0,
    )

    if "PneumoniaMNIST" in domain_choice:
        model_path = "results/global_model_medical.pt"
        if not os.path.exists(model_path):
            st.info("ℹ️ Model not yet trained for this domain — run the pipeline first.")
            return

        st.markdown("### PneumoniaMNIST Binary Classification")
        try:
            import torch
            from PIL import Image
            from src.model import MedicalCNN
            from src.dataset import get_client_dataloaders

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MedicalCNN(in_channels=1, num_classes=2).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            col_input, col_pred = st.columns([1, 1])

            with col_input:
                input_mode = st.radio("Select Input Source:", ["Bundled Real Examples", "Upload Grayscale Image"], key="p_mode")
                selected_img_tensor = None
                img_display = None
                true_label_str = "Unknown"

                if input_mode == "Bundled Real Examples":
                    _, _, test_loader, _ = get_client_dataloaders(num_clients=1, batch_size=64)
                    test_imgs, test_lbls = next(iter(test_loader))

                    ex_idx = st.selectbox(
                        "Pick Example Test Image:",
                        options=list(range(min(10, len(test_imgs)))),
                        format_func=lambda i: f"Test Sample #{i+1} (Ground Truth: {'Pneumonia' if test_lbls[i].item() == 1 else 'Normal'})",
                        key="p_ex",
                    )
                    selected_img_tensor = test_imgs[ex_idx].unsqueeze(0)
                    true_label_str = "Pneumonia" if test_lbls[ex_idx].item() == 1 else "Normal"

                    # Convert tensor to PIL for display using nearest-neighbor
                    img_np = (test_imgs[ex_idx].squeeze().numpy() * 0.5 + 0.5) * 255.0
                    img_display = Image.fromarray(img_np.astype(np.uint8)).resize((224, 224), Image.NEAREST)

                else:
                    uploaded_file = st.file_uploader("Upload Grayscale Image (28x28 or resized):", type=["png", "jpg", "jpeg"], key="p_up")
                    if uploaded_file is not None:
                        pil_img = Image.open(uploaded_file).convert("L")
                        img_display = pil_img.resize((224, 224), Image.NEAREST)

                        img_28 = pil_img.resize((28, 28), Image.BILINEAR)
                        img_arr = np.array(img_28, dtype=np.float32) / 255.0
                        img_norm = (img_arr - 0.5) / 0.5
                        selected_img_tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                if img_display is not None:
                    st.image(img_display, caption="Target Grayscale Image (28x28)", width=224)

            with col_pred:
                if selected_img_tensor is not None:
                    with torch.no_grad():
                        out = model(selected_img_tensor.to(device))
                        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                        pred_class = int(np.argmax(probs))
                        class_names = ["Normal", "Pneumonia"]

                    st.markdown("#### Inference Prediction Results")
                    pred_name = class_names[pred_class]
                    confidence = probs[pred_class] * 100.0

                    if pred_class == 1:
                        st.error(f"🚨 **Predicted Class:** `{pred_name}` ({confidence:.1f}% Confidence)")
                    else:
                        st.success(f"✅ **Predicted Class:** `{pred_name}` ({confidence:.1f}% Confidence)")

                    st.caption("Note: this model's confidence scores are not calibrated — high confidence does not guarantee correctness.")

                    if true_label_str != "Unknown":
                        st.caption(f"**Ground Truth Label:** `{true_label_str}`")

                    st.markdown("##### Class Probability Distribution")
                    st.progress(float(probs[0]), text=f"Normal: {probs[0]*100:.1f}%")
                    st.progress(float(probs[1]), text=f"Pneumonia: {probs[1]*100:.1f}%")

        except Exception as ex:
            st.error(f"Error executing PneumoniaMNIST inference: {ex}")

    elif "UCI HAR" in domain_choice:
        model_path = "results/global_model_har.pt"
        if not os.path.exists(model_path):
            st.info("ℹ️ Model not yet trained for this domain — run the pipeline first.")
            return

        st.markdown("### UCI HAR Wearable Sensor Activity Recognition")
        try:
            import torch
            from src.model_har import HARClassifier
            from src.dataset_har import get_har_client_dataloaders

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = HARClassifier(input_dim=561, num_classes=6).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            _, _, test_loader, _ = get_har_client_dataloaders(num_clients=1, batch_size=100, seed=42)
            test_X, test_y = next(iter(test_loader))

            activity_names = [
                "Walking", "Walking Upstairs", "Walking Downstairs",
                "Sitting", "Standing", "Laying"
            ]

            ex_idx = st.selectbox(
                "Select Test Sensor Record (561 Features):",
                options=list(range(min(20, len(test_X)))),
                format_func=lambda i: f"Sensor Record #{i+1} (Ground Truth: {activity_names[test_y[i].item()]})",
                key="har_ex",
            )

            feat_vector = test_X[ex_idx]
            actual_class_id = test_y[ex_idx].item()

            c_h1, c_h2 = st.columns([1, 1])

            with c_h1:
                st.markdown("#### Feature Summary Stats (561 Signals)")
                f_np = feat_vector.numpy()
                st.json({
                    "Mean Signal": float(np.mean(f_np)),
                    "Std Dev": float(np.std(f_np)),
                    "Min Signal": float(np.min(f_np)),
                    "Max Signal": float(np.max(f_np)),
                    "First 5 Signals": [round(float(v), 4) for v in f_np[:5]],
                })

            with c_h2:
                with torch.no_grad():
                    out = model(feat_vector.unsqueeze(0).to(device))
                    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                    pred_class_id = int(np.argmax(probs))

                st.markdown("#### Activity Prediction Result")
                pred_act = activity_names[pred_class_id]
                actual_act = activity_names[actual_class_id]
                conf = probs[pred_class_id] * 100.0

                if pred_class_id == actual_class_id:
                    st.success(f"✅ **Predicted Activity:** `{pred_act}` ({conf:.1f}% Confidence)")
                else:
                    st.warning(f"⚠️ **Predicted Activity:** `{pred_act}` ({conf:.1f}% Confidence)")
                st.caption(f"**Actual Ground Truth Activity:** `{actual_act}`")

                st.markdown("##### Activity Probability Distribution")
                for act_i, act_n in enumerate(activity_names):
                    p_val = float(probs[act_i])
                    st.progress(p_val, text=f"{act_n}: {p_val*100:.1f}%")

        except Exception as ex:
            st.error(f"Error executing UCI HAR inference: {ex}")

    elif "NIH Chest X-ray" in domain_choice:
        model_path = "results/global_model_nih.pt"
        if not os.path.exists(model_path):
            st.info("ℹ️ Model not yet trained for this domain — run the pipeline first.")
            return

        st.markdown("### NIH Chest X-ray Multi-Label Finding Recognition (14 Diseases)")

        # Check if NIH model failed health check
        health_warning = False
        if os.path.exists("results/phase6_nih.json"):
            try:
                with open("results/phase6_nih.json", "r") as f:
                    nih_res = json.load(f)
                if nih_res.get("status") == "aborted_not_learning":
                    health_warning = True
                else:
                    mf1 = nih_res.get("micro_f1", [0.0])[-1]
                    t_floor = nih_res.get("trivial_baseline_f1", 0.0)
                    if mf1 <= (t_floor + 0.02):
                        health_warning = True
            except Exception:
                pass

        if health_warning:
            st.warning("⚠️ **This model did not pass its training health check — predictions may be unreliable.**")

        try:
            import torch
            from PIL import Image
            from src.model_nih import NIHChestCNN
            from src.dataset_nih import get_nih_client_dataloaders, NIH_LABELS

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = NIHChestCNN(in_channels=1, num_classes=14).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            col_input, col_pred = st.columns([1, 1])

            with col_input:
                input_mode = st.radio("Select Input Source:", ["Bundled Real Examples", "Upload Grayscale Image"], key="nih_mode")
                selected_img_tensor = None
                img_display = None
                gt_findings = []

                if input_mode == "Bundled Real Examples":
                    res = get_nih_client_dataloaders(num_clients=1, batch_size=32, seed=42)
                    test_loader = res[2]
                    test_imgs, test_lbls = next(iter(test_loader))

                    ex_idx = st.selectbox(
                        "Pick Example NIH Test Image:",
                        options=list(range(min(10, len(test_imgs)))),
                        format_func=lambda i: f"Test X-ray #{i+1}",
                        key="nih_ex",
                    )
                    selected_img_tensor = test_imgs[ex_idx].unsqueeze(0)
                    t_lbl_arr = test_lbls[ex_idx].numpy()
                    gt_findings = [NIH_LABELS[k] for k in range(14) if t_lbl_arr[k] == 1.0]

                    img_np = test_imgs[ex_idx].squeeze().numpy() * 255.0
                    img_display = Image.fromarray(img_np.astype(np.uint8)).resize((256, 256), Image.NEAREST)

                else:
                    uploaded_file = st.file_uploader("Upload Chest X-ray Image (64x64 or resized):", type=["png", "jpg", "jpeg"], key="nih_up")
                    if uploaded_file is not None:
                        pil_img = Image.open(uploaded_file).convert("L")
                        img_display = pil_img.resize((256, 256), Image.NEAREST)

                        img_64 = pil_img.resize((64, 64), Image.BILINEAR)
                        img_arr = np.array(img_64, dtype=np.float32) / 255.0
                        selected_img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                if img_display is not None:
                    st.image(img_display, caption="NIH Chest X-ray Image (64x64)", width=256)

            with col_pred:
                if selected_img_tensor is not None:
                    with torch.no_grad():
                        out = model(selected_img_tensor.to(device))
                        probs = torch.sigmoid(out).cpu().numpy()[0]

                    st.markdown("#### Predicted Disease Findings (Threshold = 0.5)")

                    active_findings = [NIH_LABELS[k] for k in range(14) if probs[k] >= 0.5]
                    if active_findings:
                        st.error(f"🚨 **Findings Crossing 0.5 Threshold:** {', '.join(active_findings)}")
                    else:
                        st.success("✅ **No Finding detected (All disease probabilities < 0.5)**")

                    if gt_findings:
                        st.caption(f"**Ground Truth Findings:** {', '.join(gt_findings) if gt_findings else 'No Finding'}")

                    st.markdown("##### All 14 Finding Probabilities")
                    for k, dis_name in enumerate(NIH_LABELS):
                        p_val = float(probs[k])
                        label_txt = f"{'🚨 ' if p_val >= 0.5 else ''}{dis_name}: {p_val*100:.1f}%"
                        st.progress(p_val, text=label_txt)

        except Exception as ex:
            st.error(f"Error executing NIH Chest X-ray inference: {ex}")


# =============================================================================
# ADMIN VIEW DASHBOARD
# =============================================================================
if is_admin:
    tab_train, tab_live, tab_marketplace, tab_blockchain, tab_nacs, tab_try_model = st.tabs(
        [
            "🚀 Train Pipeline",
            "📈 Live Training Metrics",
            "🏬 Hospital Marketplace",
            "⛓️ Blockchain Ledger",
            "📊 NACS Scoring Analysis",
            "🎯 Try the Model",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: TRAIN PIPELINE
    # -------------------------------------------------------------------------
    with tab_train:
        st.subheader("Launch & Monitor FL Pipeline Runs")
        running = train_runner.is_running()

        col_cfg, col_status = st.columns([1, 1])

        with col_cfg:
            st.markdown("### Experiment Setup")

            exp_type = st.selectbox(
                "Select Experiment Pipeline Stage:",
                [
                    "Phase 1 - FL Baseline",
                    "Phase 2 - Differential Privacy",
                    "Phase 3 - Blockchain Connected",
                    "Phase 4 - NACS (single seed)",
                    "Phase 4 - NACS Multi-Seed Robustness (recommended)",
                    "Phase 5 - UCI HAR Cross-Domain Validation",
                    "Phase 6 - NIH Chest X-ray Multi-Label FL",
                ],
                index=4,
                disabled=running,
            )

            descriptions = {
                "Phase 1 - FL Baseline": "Plain FedAvg baseline across 5 hospital clients without privacy or blockchain (~30 sec).",
                "Phase 2 - Differential Privacy": "DP-SGD training with Opacus under custom target privacy epsilon (~45 sec).",
                "Phase 3 - Blockchain Connected": "DP FL logged on-chain (SHA-256 update hashes + baseline reward scores) (~1 min).",
                "Phase 4 - NACS (single seed)": "NACS score-of-record evaluation on-chain across heterogeneous privacy budgets (~1.5 min).",
                "Phase 4 - NACS Multi-Seed Robustness (recommended)": "Multi-seed benchmark (5 seeds) computing mean +/- std score variance reduction (~4-5 min).",
                "Phase 5 - UCI HAR Cross-Domain Validation": "Validates generalizability on smartphone sensor data across 5 real subjects (~2 min).",
                "Phase 6 - NIH Chest X-ray Multi-Label FL": "Validation on clinical 14-disease multi-label chest X-rays (~3 min).",
            }
            st.info(descriptions[exp_type])

            rounds = st.slider("FL Rounds", min_value=1, max_value=20, value=10, disabled=running)

            epsilon = 8.0
            if "Phase 2" in exp_type or "Phase 3" in exp_type:
                epsilon = st.number_input("Target Privacy Epsilon (ε)", min_value=0.5, max_value=50.0, value=8.0, step=0.5, disabled=running)

            seeds = 5
            if "Multi-Seed" in exp_type or "Phase 5" in exp_type:
                seeds = st.number_input("Number of Random Seeds", min_value=1, max_value=10, value=5, step=1, disabled=running)

            requires_blockchain = "Phase 3" in exp_type or "Phase 4" in exp_type or "Phase 5" in exp_type or "Phase 6" in exp_type
            if requires_blockchain and not node_connected:
                st.warning(
                    "⚠️ Local Hardhat Ethereum Node is not reachable at http://127.0.0.1:8545.\n"
                    "Off-chain logging fallback will be used if launched without Hardhat."
                )

            start_disabled = running

            if not running:
                if st.button("🚀 Start Training Run", type="primary", disabled=start_disabled):
                    if exp_type == "Phase 1 - FL Baseline":
                        train_runner.start_run("run_phase1.py", ["--rounds", str(rounds)])
                    elif exp_type == "Phase 2 - Differential Privacy":
                        train_runner.start_run("run_phase2_dp.py", ["--epsilon", str(epsilon), "--rounds", str(rounds)])
                    elif exp_type == "Phase 3 - Blockchain Connected":
                        train_runner.start_run("run_phase3_connected.py", ["--epsilon", str(epsilon), "--rounds", str(rounds)])
                    elif exp_type == "Phase 4 - NACS (single seed)":
                        train_runner.start_run("run_phase4_nacs.py", ["--rounds", str(rounds)])
                    elif "Phase 4 - NACS Multi-Seed" in exp_type:
                        train_runner.start_run("run_phase4_multiseed.py", ["--seeds", str(seeds), "--rounds", str(rounds)])
                    elif "Phase 5" in exp_type:
                        train_runner.start_run("run_phase5_har.py", ["--seeds", str(seeds), "--rounds", str(rounds)])
                    elif "Phase 6" in exp_type:
                        train_runner.start_run("run_phase6_nih.py", ["--rounds", str(rounds)])
                    st.rerun()

        with col_status:
            st.markdown("### Process & Execution Status")

            if os.path.exists("results/live_status.json"):
                try:
                    with open("results/live_status.json", "r") as f:
                        ls = json.load(f)
                    if ls.get("status") == "aborted_not_learning":
                        st.error(
                            f"🚨 **TRAINING ABORTED BY AUTOMATIC HEALTH CHECK (NOT LEARNING)**\n\n"
                            f"{ls.get('diagnostic_message', 'Model detected stalling at chance-level accuracy and loss.')}"
                        )
                except Exception:
                    pass

            if running:
                st.warning("🔄 **Training in Progress...**")
                lock_info = train_runner.get_lock_info()
                if lock_info:
                    st.caption(f"PID: `{lock_info.get('pid')}` | Script: `{lock_info.get('script_name')}`")

                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    if st.button("🛑 Stop Training", type="secondary"):
                        train_runner.stop_run()
                        st.rerun()
                with col_btn2:
                    if st.button("🔄 Refresh Output"):
                        st.rerun()
            else:
                st.success("✅ **Idle — Ready for next experiment run.**")

            st.markdown("#### Live Console Output Log (Tail)")
            log_text = train_runner.tail_log(35)
            st.code(log_text, language="text")

    # -------------------------------------------------------------------------
    # TAB 2: LIVE TRAINING METRICS
    # -------------------------------------------------------------------------
    with tab_live:
        st.subheader("Global Training Progress & Metrics Progression")

        metrics_file = None
        for p in ["results/phase6_nih.json", "results/phase4_nacs.json", "results/phase3_connected.json", "results/phase1_baseline.json"]:
            if os.path.exists(p):
                metrics_file = p
                break

        if metrics_file:
            with open(metrics_file, "r") as f:
                m_data = json.load(f)

            st.caption(f"Displaying global metrics from `{metrics_file}`")

            rounds = m_data.get("rounds", [])
            acc = m_data.get("accuracy", m_data.get("micro_f1", []))
            f1 = m_data.get("f1", m_data.get("micro_f1", []))
            loss = m_data.get("loss", [])

            df_metrics = pd.DataFrame({
                "Round": rounds,
                "Accuracy / Micro-F1": acc,
                "F1 Score": f1,
                "Loss": loss,
            }).set_index("Round")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Global Model Accuracy / F1 Progression")
                st.line_chart(df_metrics[["Accuracy / Micro-F1", "F1 Score"]])
            with c2:
                st.markdown("#### Global Model Loss Progression")
                st.line_chart(df_metrics[["Loss"]])

            st.dataframe(df_metrics.T, use_container_width=True)
        else:
            st.info("No active or completed training data available. Start a run in the Train tab.")

    # -------------------------------------------------------------------------
    # TAB 3: HOSPITAL MARKETPLACE (ADMIN VIEW)
    # -------------------------------------------------------------------------
    with tab_marketplace:
        st.subheader("Hospital Client Marketplace & Contribution Leaderboard (Admin View)")
        st.markdown(r"Full administrative view showing dataset sizes, privacy budgets ($\epsilon$), and adaptive NACS contribution scores across all 5 hospital clients.")

        df_market = pd.DataFrame({
            "Client ID": [f"Hospital {i+1}" for i in range(5)],
            "Sample Size": hospital_sizes,
            "Target Epsilon (ε)": hospital_epsilons,
            "Latest NACS Score": [hospital_latest_scores[i] for i in range(5)],
            "Rank": [hospital_ranks[i] for i in range(5)],
        }).sort_values("Rank")

        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.metric("Total Participating Hospitals", "5 Hospitals")
        with m_c2:
            st.metric("Total Dataset Samples", f"{sum(hospital_sizes):,} Samples")
        with m_c3:
            st.metric("Top NACS Contribution Score", f"{max(hospital_latest_scores.values())} / 100")

        st.markdown("### Client Contribution Leaderboard")
        st.dataframe(df_market, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 4: BLOCKCHAIN LEDGER (ADMIN VIEW)
    # -------------------------------------------------------------------------
    with tab_blockchain:
        st.subheader("Smart Contract ContributionLedger Audit Inspection (Full Admin Ledger)")

        if node_connected:
            st.success("🟢 Connected to Hardhat Ethereum Node at `http://127.0.0.1:8545`")
            try:
                from src.blockchain_client import BlockchainLogger
                logger = BlockchainLogger()

                st.markdown(f"**Contract Address:** `{logger.contract_address}` | **Account #0:** `{logger.account}`")

                rows = []
                for cid in range(5):
                    hist = logger.get_client_history(cid)
                    for r_up, h_up, s_rew in zip(
                        hist["update_rounds"],
                        hist["update_hashes"],
                        hist["reward_scores"],
                    ):
                        rows.append({
                            "Client ID": f"Hospital {cid + 1}",
                            "Round": r_up,
                            "Weight SHA-256 Hash": f"{h_up[:12]}...{h_up[-8:]}",
                            "Logged NACS Score": s_rew,
                        })

                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No transaction history recorded on smart contract yet.")
            except Exception as ex:
                st.error(f"Error querying smart contract: {ex}")
        else:
            st.warning("⚠️ Hardhat node is disconnected. Displaying cached on-chain log records from local JSON.")
            if nacs_json_data:
                bc_log = nacs_json_data.get("blockchain_log", [])
                if bc_log:
                    st.dataframe(pd.DataFrame(bc_log), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 5: NACS SCORING ANALYSIS (ADMIN VIEW)
    # -------------------------------------------------------------------------
    with tab_nacs:
        st.subheader("NACS Contribution Score Variance & Robustness Benchmark")

        has_multiseed = os.path.exists("results/phase4_nacs_multiseed.json")
        has_singleseed = os.path.exists("results/phase4_nacs.json")

        if has_multiseed:
            st.success("✅ **Multi-Seed Robustness Benchmark Results (5 Seeds)**")
            with open("results/phase4_nacs_multiseed.json", "r") as f:
                ms_data = json.load(f)

            client_dict = ms_data.get("client_summary", ms_data.get("per_client", ms_data))
            rows = []
            for cid in sorted([int(k) for k in client_dict.keys() if str(k).isdigit()]):
                info = client_dict[str(cid)]
                m_red = info.get("reduction_mean_percent", info.get("mean_variance_reduction", 0.0))
                s_red = info.get("reduction_std_percent", info.get("std_variance_reduction", 0.0))
                rows.append({
                    "Client": f"Hospital {cid + 1}",
                    "Target Epsilon (ε)": info.get("target_epsilon", "N/A"),
                    "Mean Variance Reduction (%)": f"{m_red:.2f}%",
                    "Std Variance Reduction (%)": f"± {s_red:.2f}%",
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if os.path.exists("results/phase4_multiseed_comparison.png"):
                st.image("results/phase4_multiseed_comparison.png", caption="Mean Score Variance Reduction Across Random Seeds")
        elif has_singleseed:
            st.info("Displaying Single-Seed NACS Variance Reduction")
            if os.path.exists("results/phase4_nacs_comparison.png"):
                st.image("results/phase4_nacs_comparison.png", caption="Baseline vs. NACS Score Progressions")

    # -------------------------------------------------------------------------
    # TAB 6: TRY THE MODEL (ADMIN VIEW)
    # -------------------------------------------------------------------------
    with tab_try_model:
        render_try_the_model_tab()


# =============================================================================
# HOSPITAL PARTICIPANT VIEW DASHBOARD (SIMULATED HOSPITAL WORKSPACE)
# =============================================================================
else:
    cid = selected_hospital_id
    h_name = f"Hospital {cid + 1}"
    h_eps = hospital_epsilons[cid]
    h_size = hospital_sizes[cid]
    h_rank = hospital_ranks.get(cid, 1)
    h_score = hospital_latest_scores.get(cid, 50)

    # Participant View Banner
    col_b1, col_b2 = st.columns([3, 1])
    with col_b1:
        st.markdown(f"### 🏥 Participant Workspace: **{h_name}**")
        st.caption("Secure hospital node interface. View your dataset, local performance metrics, NACS score history, and audit ledger.")
    with col_b2:
        st.markdown(
            f"""
            <div class="rank-card">
                <div class="rank-title">Your Current Rank</div>
                <div class="rank-value">#{h_rank} <span style="font-size:1.1rem;font-weight:400;">/ 5</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Privacy Guarantee Info Box
    st.info(
        rf"🔒 **Privacy Guarantee:** You are viewing **{h_name}'s private participant dashboard**. "
        r"Competitor hospitals' raw NACS scores, dataset sizes, and privacy budgets ($\epsilon$) are masked."
    )

    # Key Participant Metrics Summary Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-box'><b>Local Dataset Size</b><br><h2>{h_size:,} samples</h2></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-box'><b>Target Privacy Budget (ε)</b><br><h2>ε = {h_eps}</h2></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-box'><b>Latest NACS Score</b><br><h2>{h_score} / 100</h2></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric-box'><b>Current Leaderboard Rank</b><br><h2>#{h_rank} of 5</h2></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Participant Tabs
    htab_history, htab_blockchain, htab_controls, htab_export, htab_try_model = st.tabs(
        [
            "📈 My Training & NACS History",
            "⛓️ My Blockchain Audit Records",
            "⚙️ Next-Round & What-If Controls",
            "📥 Export My Report",
            "🎯 Try the Model",
        ]
    )

    # Extract per-client history for selected hospital
    client_history_rows = []
    if nacs_json_data and "per_client" in nacs_json_data:
        c_data = nacs_json_data["per_client"].get(str(cid), {})
        c_hist = c_data.get("history", [])
        for entry in c_hist:
            client_history_rows.append({
                "Round": entry.get("round", 0),
                "NACS Score": entry.get("nacs_score", 0),
                "Baseline Score": entry.get("baseline_score", 0),
                "Noise Multiplier (σ)": round(entry.get("noise_multiplier", 0.0), 4),
                "Epsilon (ε)": entry.get("target_epsilon", h_eps),
                "Loss Before": round(entry.get("loss_before", 0.0), 4),
                "Loss After": round(entry.get("loss_after", 0.0), 4),
            })

    # -------------------------------------------------------------------------
    # PARTICIPANT TAB 1: MY TRAINING & NACS HISTORY
    # -------------------------------------------------------------------------
    with htab_history:
        st.subheader(f"Round-by-Round Contribution History ({h_name})")

        if client_history_rows:
            df_chist = pd.DataFrame(client_history_rows)

            col_chart, col_tbl = st.columns([1, 1])

            with col_chart:
                st.markdown("#### NACS Score vs. Noise-Blind Baseline Evolution")
                fig, ax = plt.subplots(figsize=(6, 3.8))
                ax.plot(df_chist["Round"], df_chist["Baseline Score"], "o--", color="#EF4444", linewidth=2, label="Baseline (Noise-Blind)")
                ax.plot(df_chist["Round"], df_chist["NACS Score"], "s-", color="#2563EB", linewidth=2.5, label="NACS (Noise-Aware)")
                ax.set_xlabel("FL Round", fontsize=10, fontweight="bold")
                ax.set_ylabel("Contribution Score (0-100)", fontsize=10, fontweight="bold")
                ax.set_ylim(-5, 105)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(loc="upper right", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)

            with col_tbl:
                st.markdown("#### Audit Record Table")
                st.dataframe(df_chist, use_container_width=True, hide_index=True)
        else:
            st.info("No training history recorded for this hospital client yet. Launch a run in the Train tab.")

    # -------------------------------------------------------------------------
    # PARTICIPANT TAB 2: MY BLOCKCHAIN AUDIT RECORDS
    # -------------------------------------------------------------------------
    with htab_blockchain:
        st.subheader(f"Auditable Blockchain Record Ledger ({h_name})")
        st.markdown("Filtered on-chain ledger records belonging strictly to your hospital node.")

        my_bc_rows = []
        if node_connected:
            try:
                from src.blockchain_client import BlockchainLogger
                logger = BlockchainLogger()
                hist = logger.get_client_history(cid)
                for r_up, h_up, s_rew in zip(
                    hist["update_rounds"],
                    hist["update_hashes"],
                    hist["reward_scores"],
                ):
                    my_bc_rows.append({
                        "Hospital ID": h_name,
                        "FL Round": r_up,
                        "SHA-256 Weight Hash": h_up,
                        "On-Chain NACS Reward Score": s_rew,
                        "Status": "Verified On-Chain",
                    })
            except Exception as ex:
                st.caption(f"Note querying contract: {ex}")

        if not my_bc_rows and nacs_json_data:
            bc_log = nacs_json_data.get("blockchain_log", [])
            for log_entry in bc_log:
                if log_entry.get("client_id") == cid:
                    my_bc_rows.append({
                        "Hospital ID": h_name,
                        "FL Round": log_entry.get("round"),
                        "Update Tx Hash / Off-Chain": log_entry.get("update_tx_hash", "off-chain"),
                        "On-Chain NACS Reward Score": log_entry.get("nacs_score"),
                        "Reward Tx Hash": log_entry.get("reward_tx_hash", "off-chain"),
                    })

        if my_bc_rows:
            st.dataframe(pd.DataFrame(my_bc_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No blockchain transaction records found for your hospital node yet.")

    # -------------------------------------------------------------------------
    # PARTICIPANT TAB 3: NEXT-ROUND & WHAT-IF CONTROLS
    # -------------------------------------------------------------------------
    with htab_controls:
        st.subheader(f"Future Round Privacy & Contribution Configuration ({h_name})")
        st.caption("Configure privacy settings and data contribution preferences for future FL rounds, or run hypothetical what-if preview simulations.")

        c_ctrl1, c_ctrl2 = st.columns([1, 1])

        with c_ctrl1:
            st.markdown("#### 1. Next-Round Privacy Budget Setting")
            next_eps = st.select_slider(
                "Configure Next Round Privacy Budget (ε):",
                options=[1.0, 3.0, 8.0, 15.0, 20.0],
                value=float(h_eps),
                help="Select target privacy budget for future training rounds.",
            )

            if next_eps <= 3.0:
                st.warning(f"🔒 **Strong Privacy Selected (ε = {next_eps}):** High DP noise (σ) injected. Protects raw data strongly, but may reduce local model weight precision.")
            elif next_eps <= 8.0:
                st.info(f"⚖️ **Balanced Privacy Selected (ε = {next_eps}):** Moderate DP noise. Optimal trade-off between privacy guarantee and model accuracy.")
            else:
                st.success(f"⚡ **High Utility Selected (ε = {next_eps}):** Low DP noise. Maximizes model precision, but provides lower formal privacy guarantee.")

            st.markdown("---")
            st.markdown("#### 2. Future Round Participation Toggle")
            part_next = st.toggle("Participate in Next FL Round", value=True)
            if not part_next:
                st.error(f"⚠️ **{h_name} will opt out** of the next configured FL round. Local weights will not be aggregated.")
            else:
                st.success(f"✅ **{h_name} will participate** in the next configured FL round.")

        with c_ctrl2:
            st.markdown("#### 3. Data Contribution Fraction Control")
            data_fraction = st.slider(
                "Local Data Contribution Fraction:",
                min_value=25,
                max_value=100,
                value=100,
                step=25,
                format="%d%%",
                help="Simulate contributing a subset of your hospital's dataset.",
            )
            sim_samples = int(h_size * (data_fraction / 100.0))
            st.markdown(f"**Selected Contribution:** `{data_fraction}%` $\\rightarrow$ **~{sim_samples:,} samples** (out of {h_size:,} available)")

            st.markdown("---")

            # What-If NACS Preview Section
            st.markdown("#### 4. What-If NACS Preview Simulation")
            st.caption("Estimates hypothetical NACS contribution score based on your selected future parameters.")

            # Lightweight preview math: baseline size score vs NACS noise-adjusted preview
            quality_factor = 0.85  # Stored quality estimate
            sample_ratio = sim_samples / sum(hospital_sizes)
            noise_ratio = 1.0 / max(0.5, np.sqrt(next_eps))

            # NACS multi-factor formula simulation
            preview_size_score = int(round(sample_ratio * 100 * 3.5))
            preview_nacs_score = int(round((sample_ratio * 0.4 + quality_factor * 0.4 + (1.0 / (1.0 + noise_ratio)) * 0.2) * 85))
            preview_nacs_score = min(100, max(5, preview_nacs_score))

            st.markdown(
                f"""
                <div class="preview-box">
                    <b>Hypothetical Score Preview for {h_name}:</b><br>
                    • Size-Only Baseline Score: <b>{preview_size_score} / 100</b><br>
                    • <b>NACS (Noise-Aware) Score: <span style="color:#2563EB;font-size:1.3rem;">{preview_nacs_score} / 100</span></b><br><br>
                    <small>⚠️ <i>Hypothetical preview — no model training performed.</i></small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # PARTICIPANT TAB 4: EXPORT MY REPORT
    # -------------------------------------------------------------------------
    with htab_export:
        st.subheader(f"Export Participant Report ({h_name})")
        st.markdown("Download a clean CSV report containing only your hospital's historical metrics, NACS scores, and audit records.")

        if client_history_rows:
            df_export = pd.DataFrame(client_history_rows)
            df_export["Hospital ID"] = h_name
            df_export["Current Rank"] = f"#{h_rank} of 5"

            csv_data = df_export.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {h_name} Audit Report (CSV)",
                data=csv_data,
                file_name=f"{h_name.lower().replace(' ', '_')}_report.csv",
                mime="text/csv",
                type="primary",
            )
            st.dataframe(df_export, use_container_width=True, hide_index=True)
        else:
            st.info("No data available to export yet. Complete a training run first.")

    # -------------------------------------------------------------------------
    # PARTICIPANT TAB 5: TRY THE MODEL
    # -------------------------------------------------------------------------
    with htab_try_model:
        render_try_the_model_tab()
