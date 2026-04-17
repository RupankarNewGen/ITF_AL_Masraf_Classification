import streamlit as st
import json
import os

st.set_page_config(layout="wide", page_title="Manual Validation UI - Multi User")

MANIFEST_PATH = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/Data/input_jsons/lc_drawing_remaining_images.json"
RESULTS_DIR = "/datadrive2/IDF_AL_MASRAF/LC_Drawing_Full_Result"
VALIDATION_SAVE_PATH_V1 = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/validation_results_validator1.json"
VALIDATION_SAVE_PATH_V2 = "/datadrive2/IDF_AL_MASRAF/ITF_Al_Masraf_Classification/validation_results_validator2.json"

VALID_CLASSES = [
    "Covering_Schedule",
    "Bill_of_Exchange",
    "Commercial_Invoice",
    "Invoice",
    "Packing_List",
    "Bill_of_Lading",
    "Airway_Bill",
    "Sea_Waybill",
    "Charter_Party_Bill",
    "Multimodal_Transport_Document",
    "ROAD-RAIL-INLAND-WATERWAY_TRANSPORT",
    "Certificate_of_Origin",
    "Insurance",
    "Phytosnaitary_Certificate",
    "Inspection_Certificate",
    "Seaworthy_Vessel_Certificate",
    "Beneficiary_Certificate",
    "Quality_Certificate",
    "Weight_Certificate",
    "Fumigation_Certificate",
    "Delivery Notes",
    "LC",
    "Others",
    "Doubt"
]

@st.cache_data
def load_manifest():
    with open(MANIFEST_PATH, 'r') as f:
        return json.load(f)

def load_validated_paths(save_path):
    if not os.path.exists(save_path):
        return set()
    validated = set()
    with open(save_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    validated.add(data['image_path'])
                except:
                    pass
    return validated

def save_validation(save_path, image_path, original_pred, validated_class, status):
    with open(save_path, 'a') as f:
        record = {
            "image_path": image_path,
            "original_prediction": original_pred,
            "validated_class": validated_class,
            "status": status
        }
        f.write(json.dumps(record) + "\n")

st.title("Document Classification Validation Queue")

# Top Level Profile Selection
st.sidebar.title("Configuration")
validator_profile = st.sidebar.selectbox("Select Validator Profile", ["Validator 1", "Validator 2"], help="Selecting your profile automatically assigns you exactly half of the dataset.")

manifest_full = load_manifest()
midpoint = len(manifest_full) // 2

# Assign Split
if validator_profile == "Validator 1":
    manifest = manifest_full[:midpoint]
    save_path = VALIDATION_SAVE_PATH_V1
else:
    manifest = manifest_full[midpoint:]
    save_path = VALIDATION_SAVE_PATH_V2

# Load validations
validated_paths = load_validated_paths(save_path)

# Determine the absolute first unvalidated image
start_idx = 0
for i, path in enumerate(manifest):
    if path not in validated_paths:
        start_idx = i
        break
else:
    start_idx = len(manifest)

# Initialize or reset session logic
if 'profile' not in st.session_state or st.session_state.profile != validator_profile:
    st.session_state.profile = validator_profile
    st.session_state.current_index = start_idx
# REMOVED the strict current_index < start_idx override here to allow "Previous" backward navigation

if st.session_state.current_index >= len(manifest):
    st.success(f"**{validator_profile}** has completely finished their half of the dataset! Amazing work.")
    # Allow going back even if finished
    if st.button("⬅️ Go Back to Previous Image"):
        st.session_state.current_index -= 1
        st.rerun()
    st.stop()

# Queue Stats
total_assigned = len(manifest)
completed = len(validated_paths)
st.sidebar.markdown("---")
st.sidebar.metric(label="Your Assigned Queue", value=total_assigned)
st.sidebar.metric(label="Completed", value=completed)
st.sidebar.metric(label="Remaining", value=total_assigned - completed)
st.sidebar.progress(completed / total_assigned if total_assigned > 0 else 1.0)

current_image_path = manifest[st.session_state.current_index]
filename = os.path.basename(current_image_path)
json_filename = filename.rsplit('.', 1)[0] + '.json'
json_path = os.path.join(RESULTS_DIR, json_filename)

st.write(f"**Current Index:** {st.session_state.current_index + 1} of {total_assigned}")

col1, col2 = st.columns([2, 1])

with col1:
    if os.path.exists(current_image_path):
        st.image(current_image_path, use_container_width=True)
    else:
        st.error(f"Image not found: {current_image_path}")

with col2:
    st.markdown("### Prediction Details")
    st.write(f"**File:** `{filename}`")
    
    pred_class = "Unknown"
    reasoning = "N/A"
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
                pred_class = data.get("classification", "Unknown")
                reasoning = data.get("reasoning", "N/A")
            except:
                pass
    else:
        st.warning(f"Prediction JSON not found for this image: {json_filename}")
        
    st.info(f"**Model Predicted Class:** {pred_class}")
    st.write(f"**Reasoning:** {reasoning}")
    
    st.markdown("---")
    st.markdown("### Validation")
    
    if st.button("✅ Approve Prediction", type="primary", use_container_width=True):
        save_validation(save_path, current_image_path, pred_class, pred_class, "approved")
        st.session_state.current_index += 1
        st.rerun()
        
    st.markdown("**OR Correct the Class:**")
    
    default_idx = VALID_CLASSES.index(pred_class) if pred_class in VALID_CLASSES else 0
    selected_class = st.selectbox("Select correct class", VALID_CLASSES, index=default_idx)
    
    if st.button("🔄 Correct & Complete", use_container_width=True):
        save_validation(save_path, current_image_path, pred_class, selected_class, "corrected")
        st.session_state.current_index += 1
        st.rerun()
        
    st.markdown("---")
    if st.button("⏭️ Skip Image (Don't Save)", use_container_width=True):
        st.session_state.current_index += 1
        st.rerun()

    # --- NEW FUNCTIONALITY: PREVIOUS / NEXT NAVIGATION ---
    st.markdown("---")
    nav_col1, nav_col2 = st.columns(2)
    
    with nav_col1:
        if st.button("⬅️ Previous Image", use_container_width=True):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
            else:
                st.toast("You are already at the very first image.")
                
    with nav_col2:
        if st.button("Next Image ➡️", use_container_width=True):
            if st.session_state.current_index < len(manifest) - 1:
                st.session_state.current_index += 1
                st.rerun()
            else:
                st.toast("You are at the last image.")