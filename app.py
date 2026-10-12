import streamlit as st
import pandas as pd
import pymupdf as fitz
from PIL import Image
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyreadstat
from google import genai
from google.genai import types

# Page setup
st.set_page_config(
    page_title="OYSHIA Encounter Form Scanner to SPSS",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Automated OYSHIA Encounter Form Scanner ➡️ SPSS")
st.markdown("Upload scanned encounter forms (PDF / Images) to instantly extract patient records and download a native **SPSS (.sav)** file.")

# --- SIDEBAR: Configuration ---
st.sidebar.header("⚙️ Configuration")

default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
api_key = st.sidebar.text_input("Gemini API Key", value=default_key, type="password", help="Get a free API key at https://aistudio.google.com")

if not api_key:
    st.sidebar.warning("⚠️ Please provide a Gemini API Key to enable AI scanning.")
    st.sidebar.markdown("[👉 Get a free Gemini API Key here](https://aistudio.google.com)")

concurrency = st.sidebar.slider("Parallel Processing Speed (Pages at once)", min_value=1, max_value=8, value=4)

# --- SPSS DICTIONARY MAPS ---
LGA_MAP = {"OORELOPE": "26", "AFIJIO": "1", "AKINYELE": "2", "ATIBA": "3", "ATISBO": "4", "EGBEDA": "5", "IBADAN NORTH": "6"}
SEX_MAP = {"M": 1.0, "MALE": 1.0, "F": 2.0, "FEMALE": 2.0}
EMPLOYER_MAP = {"CIVIL SERVANT": 1.0, "R.T. CIVIL SERVANT": 1.0, "NON CIVIL SERVANT": 2.0}

DIAG_MAP = {
    "HTN": 2.0, "HYPERTENSION": 2.0, "URTI": 3.0, "MALARIA": 4.0, "SEPSIS": 8.0,
    "MSK PAIN": 13.0, "MSS PAIN": 13.0, "PUD": 22.0, "SEVERE ANEMIA": 33.0,
    "HHDX": 38.0, "DIABETES": 50.0, "DM": 50.0, "DERMATITIS": 56.0,
    "INSOMNIA": 78.0, "TENSION HEADACHE": 95.0, "MIGRAINE": 95.0, "DYSPEPSIA": 98.0,
    "LOW BACK PAIN": 101.0, "BACK PAIN": 101.0, "MALAISE": 181.0, "MFC": 184.0, "COLIC": 185.0
}

INVEST_MAP = {
    "PCV": 1.0, "MP": 2.0, "RBS": 3.0, "WBC": 4.0, "FBC": 4.0, "PT": 6.0, "HCG": 6.0,
    "URINALYSIS": 7.0, "FBG": 23.0, "FBS": 34.0, "MRDT": 35.0, "SPT": 94.0
}

TREAT_MAP = {
    "TAB IBUPROFEN": 1.0, "CAP OMEPRAZOLE": 2.0, "TAB PCM": 3.0, "TAB AMLODIPINE": 4.0,
    "TAB VASOPRIN": 5.0, "TAB ACT": 8.0, "CAP AMOXICILLIN": 9.0, "TAB MULTIVITAMIN": 10.0,
    "TAB CHYMORAL": 15.0, "TAB FLAGYL": 17.0, "TAB VITAMIN C": 23.0, "TAB LORATADINE": 24.0,
    "TAB DICLOFENAC": 25.0, "TAB NIFEDIPINE": 27.0, "TAB MODURETIC": 28.0, "TAB PREDNISOLONE": 38.0,
    "TAB ALBENDAZOLE": 39.0, "COUGH SYRUP": 44.0, "IM PCM": 45.0, "TAB VITAMIN-BCO": 50.0,
    "IM DICLOFENAC": 51.0, "TAB GESTID": 52.0, "TAB AMPICLOX": 55.0, "TAB ORPHENADRINE": 56.0,
    "GENTAMICIN EAR DROP": 61.0, "TAB AMITRIPTYLINE": 62.0, "IV NORMAL SALINE": 66.0,
    "IM PENTAZOCINE": 67.0, "IV CEFTRIAZONE": 68.0, "IV CEFTRIAXONE": 68.0, "TAB METFORMIN": 70.0,
    "TAB FERROUS": 78.0, "TAB AMOXICLAV": 80.0, "TAB DAONIL": 92.0, "TAB PIRITON": 97.0,
    "TAB FLUCONAZOLE": 99.0, "TAB DIAZEPAM": 104.0, "IV ARTESUNATE": 107.0, "TAB CIPROFLOXACIN": 140.0,
    "IM GENTAMICIN": 141.0, "TAB LASIX": 148.0, "TAB PIROXICAM": 149.0, "TAB DOXYCYCLINE": 163.0,
    "TAB PROPRANOLOL": 164.0, "TAB ALDOMET": 166.0, "TAB LISINOPRIL": 167.0, "TAB MELOXICAM": 168.0,
    "TAB BUSCOPAN": 169.0, "TAB BROMAZEPAM": 170.0, "EAR DROPS": 171.0, "TAB LOMOTIL": 172.0,
    "TAB CIMETIDINE": 173.0, "DICLOFENAC GEL": 174.0, "IVF 5% DEXTROSE": 175.0,
    "TAB AMINOPHYLLINE": 176.0, "TAB CEFUROXIME": 177.0, "IV DEXAMETHASONE": 178.0,
    "TAB SPIRONOLACTONE": 179.0, "TAB ERYTHROMYCIN": 180.0, "TAB TRANEXAMIC ACID": 182.0, "GASCOL SUSP": 183.0
}

# --- HELPER FUNCTIONS ---
def extract_images_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("jpeg")
        images.append(Image.open(io.BytesIO(img_bytes)))
    return images

def process_page_with_ai(image, page_num, client):
    prompt = """
    You are an expert medical records assistant analyzing a Nigerian OYSHIA (Oyo State Health Insurance Agency) Patients' Encounter Form.
    Extract the header info and all patient rows from this scanned form.
    
    Return a strict JSON object with this structure:
    {
      "lga": "Oorelope",
      "facility": "General Hospital, Igboho",
      "month": 10,
      "year": 2025,
      "patients": [
        {
          "sn": 1,
          "name": "Full patient name",
          "age": 45,
          "sex": "F",
          "employer": "Civil Servant",
          "oyshia_id": "66508",
          "diagnoses": ["HTN", "Diabetes"],
          "investigations": ["FBS"],
          "treatments": ["Tab Metformin", "Tab Amlodipine"],
          "phone": "08012345678"
        }
      ]
    }
    Rules:
    - Look carefully at cursive medical handwriting.
    - If empty or illegible, return null or empty list.
    """
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=85)
    img_byte_arr = img_byte_arr.getvalue()
    
    # Using the updated model gemini-3.8-flash
    response = client.models.generate_content(
        model='gemini-3.8-flash',
        contents=[
            types.Part.from_bytes(data=img_byte_arr, mime_type='image/jpeg'),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    data = json.loads(response.text)
    return page_num, data

def match_code(text, lookup_map):
    if not text:
        return None
    cleaned = str(text).upper().strip()
    for key, val in lookup_map.items():
        if key in cleaned:
            return val
    return None

# --- MAIN WORKFLOW ---
existing_spss = st.file_uploader("📂 (Optional) Upload Existing SPSS (.sav) File to Append New Records into", type=["sav"])

uploaded_file = st.file_uploader("📸 Upload Scanned Encounter Form (PDF, PNG, JPG)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file and api_key:
    client = genai.Client(api_key=api_key)
    
    with st.spinner("Reading file pages..."):
        if uploaded_file.type == "application/pdf":
            pages = extract_images_from_pdf(uploaded_file.read())
        else:
            pages = [Image.open(uploaded_file)]
    
    st.info(f"📄 Found **{len(pages)} page(s)** in the document.")
    
    if st.button("🚀 Start Parallel Scan & Extraction"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        all_results = []
        completed = 0
        total_pages = len(pages)
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_page = {
                executor.submit(process_page_with_ai, img, idx + 1, client): idx + 1 
                for idx, img in enumerate(pages)
            }
            
            for future in as_completed(future_to_page):
                page_idx = future_to_page[future]
                try:
                    p_num, page_data = future.result()
                    all_results.append((p_num, page_data))
                except Exception as e:
                    st.error(f"Error on Page {page_idx}: {e}")
                
                completed += 1
                progress_bar.progress(completed / total_pages)
                status_text.text(f"⚡ Processed {completed} of {total_pages} pages...")
        
        all_results.sort(key=lambda x: x[0])
        
        flat_records = []
        for p_num, p_data in all_results:
            header_lga = p_data.get("lga", "OORELOPE")
            header_fac = p_data.get("facility", "General Hospital, Igboho")
            header_month = float(p_data.get("month", 10.0))
            header_year = float(p_data.get("year", 2025.0))
            
            for pat in p_data.get("patients", []):
                phone = str(pat.get("phone", "")).replace(" ", "").replace("-", "")[:11]
                
                record = {
                    "Year": header_year,
                    "FacilityPHC": header_fac,
                    "LGA": match_code(header_lga, LGA_MAP) or "26",
                    "Month": header_month,
                    "Name": pat.get("name", "").strip().upper(),
                    "Age": float(pat.get("age")) if pat.get("age") else None,
                    "Sex": match_code(pat.get("sex"), SEX_MAP),
                    "Employer": match_code(pat.get("employer"), EMPLOYER_MAP) or 1.0,
                    "OYSHIA_ID": float(str(pat.get("oyshia_id", 0)).replace(",", "")) if pat.get("oyshia_id") else None,
                    "Phone_contact": phone
                }
                
                diags = pat.get("diagnoses", [])
                for i in range(1, 8):
                    val = diags[i-1] if i-1 < len(diags) else None
                    record[f"Diagnosis{i}"] = match_code(val, DIAG_MAP)
                
                invs = pat.get("investigations", [])
                for i in range(1, 8):
                    val = invs[i-1] if i-1 < len(invs) else None
                    record[f"Investigation{i}"] = match_code(val, INVEST_MAP)
                
                treats = pat.get("treatments", [])
                t_cols = ['Treatment1', 'treatment2', 'Treatment3', 'Treatment4', 'Treatment5', 'treatment6', 'Treatment7']
                for i, col_name in enumerate(t_cols):
                    val = treats[i] if i < len(treats) else None
                    record[col_name] = match_code(val, TREAT_MAP)
                
                flat_records.append(record)
        
        new_df = pd.DataFrame(flat_records)
        
        # If existing SPSS file was uploaded, append to it!
        if existing_spss:
            temp_existing_path = "temp_existing.sav"
            with open(temp_existing_path, "wb") as f:
                f.write(existing_spss.getbuffer())
            existing_df, meta = pyreadstat.read_sav(temp_existing_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            st.session_state["extracted_df"] = combined_df
            st.success(f"🎉 Appended {len(new_df)} new records to existing {len(existing_df)} records! (Total: {len(combined_df)})")
        else:
            st.session_state["extracted_df"] = new_df
            st.success(f"🎉 Successfully extracted **{len(new_df)} patient encounters** across {total_pages} pages!")

# --- EDITABLE PREVIEW & DOWNLOAD ---
if "extracted_df" in st.session_state:
    st.subheader("📊 Review & Edit Data")
    st.markdown("You can edit any cell directly in the table below before downloading:")
    
    edited_df = st.data_editor(st.session_state["extracted_df"], use_container_width=True, num_rows="dynamic")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sav_buffer = "output_dataset.sav"
        var_val_labels = {
            "Sex": {1.0: "Male", 2.0: "Female"},
            "Employer": {1.0: "Civil Servant", 2.0: "Non Civil Servant", 3.0: "Missing"},
            "Month": {float(m): f"Month {m}" for m in range(1, 13)},
            "Diagnosis1": {v: k for k, v in DIAG_MAP.items()},
            "Treatment1": {v: k for k, v in TREAT_MAP.items()}
        }
        
        pyreadstat.write_sav(
            edited_df,
            sav_buffer,
            column_labels=list(edited_df.columns),
            variable_value_labels=var_val_labels
        )
        
        with open(sav_buffer, "rb") as f:
            st.download_button(
                label="📥 Download SPSS Dataset (.sav)",
                data=f,
                file_name="ENCOUNTER_FORM_PROCESSED.sav",
                mime="application/x-spss-sav",
                use_container_width=True
            )
            
    with col2:
        excel_buffer = io.BytesIO()
        edited_df.to_excel(excel_buffer, index=False)
        st.download_button(
            label="📊 Download Excel File (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name="ENCOUNTER_FORM_PROCESSED.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )