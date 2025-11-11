# app.py - Cross-Domain Fake Review Detection System with Batch Analysis
import streamlit as st
from pathlib import Path
import json
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime

# Local project imports
try:
    from src.model import load_model, get_bert_embeddings
    from src.preprocess import clean_text
    from src.utils import MODELS_DIR, read_metadata
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="centered"
)

# ========================
# Authentication Settings
# ========================
try:
    COOKIE_NAME = st.secrets.get("cookie_name", "fake_review_cookie")
    COOKIE_KEY = st.secrets.get("cookie_key", "local_dev_change_this_to_a_strong_secret")
    COOKIE_EXPIRY = st.secrets.get("cookie_expiry_days", 30)
except Exception:
    COOKIE_NAME = "fake_review_cookie"
    COOKIE_KEY = "local_dev_change_this_to_a_strong_secret"
    COOKIE_EXPIRY = 30

# ========================
# Credentials Management
# ========================
CRED_DIR = Path("credentials")
CRED_DIR.mkdir(exist_ok=True)
CRED_PATH = CRED_DIR / "credentials.json"


def load_credentials() -> dict:
    """Load user credentials from JSON file"""
    if CRED_PATH.exists():
        try:
            with open(CRED_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {"usernames": {}}
                if "usernames" not in data:
                    data["usernames"] = {}
                return data
        except json.JSONDecodeError:
            return {"usernames": {}}
    return {"usernames": {}}


def save_credentials(creds: dict) -> bool:
    """Save user credentials to JSON file"""
    try:
        with open(CRED_PATH, "w", encoding="utf-8") as f:
            json.dump(creds, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


# Load credentials
credentials = load_credentials()

# ========================
# Session State Initialization
# ========================
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'single_review_history' not in st.session_state:
    st.session_state['single_review_history'] = []

# ========================
# Authenticator Setup
# ========================
try:
    authenticator = stauth.Authenticate(
        credentials,
        COOKIE_NAME,
        COOKIE_KEY,
        COOKIE_EXPIRY
    )
except Exception:
    st.error("Unable to initialize authentication system")
    st.stop()

# ========================
# Application Header
# ========================
st.title("Fake Review Detector")
st.markdown("Detect fake reviews using AI-powered analysis")

# ========================
# Sidebar - Clean & Informative
# ========================
with st.sidebar:
    st.markdown("### How It Works")
    st.markdown("""
    **Step 1: Enter a Review**  
    Copy and paste any product or service review
    
    **Step 2: AI Analysis**  
    Our system analyzes language patterns and authenticity markers
    
    **Step 3: Get Results**  
    Receive a confidence score showing if the review is genuine or fake
    """)
    
    st.markdown("---")
    
    # Navigation
    auth_choice = st.selectbox(
        "Account",
        ["Login", "Sign Up"],
        label_visibility="collapsed"
    )

# ========================
# SIGN UP INTERFACE - Compact 2-Column Layout
# ========================
if auth_choice == "Sign Up":
    st.markdown("### Create Your Account")
    st.markdown("Join to access the review analysis system")
    
    col1, col2 = st.columns(2)
    
    with col1:
        username = st.text_input(
            "Username",
            max_chars=50,
            placeholder="Choose a username"
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="At least 6 characters"
        )
    
    with col2:
        name = st.text_input(
            "Full Name",
            max_chars=100,
            placeholder="Your name"
        )
        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Re-enter password"
        )
    
    if st.button("Create Account", type="primary", use_container_width=True):
        # Input validation
        if not username or not name or not password:
            st.error("Please fill in all fields")
        elif " " in username:
            st.error("Username cannot contain spaces")
        elif len(username) < 3:
            st.error("Username must be at least 3 characters")
        elif password != confirm_password:
            st.error("Passwords do not match")
        elif len(password) < 6:
            st.warning("Password should be at least 6 characters")
        elif username in credentials.get("usernames", {}):
            st.error(f"Username '{username}' is already taken")
        else:
            try:
                # Hash password
                hashed_passwords = stauth.Hasher([password]).generate()
                
                # Store user
                if "usernames" not in credentials:
                    credentials["usernames"] = {}
                
                credentials["usernames"][username] = {
                    "name": name,
                    "password": hashed_passwords[0]
                }
                
                # Save to file
                if save_credentials(credentials):
                    st.success("Account created successfully")
                    st.info("Switch to 'Login' in the sidebar to sign in")
                    # Clear any session state to prevent auto-login
                    st.session_state['authentication_status'] = None
                    st.session_state['name'] = None
                    st.session_state['username'] = None
                else:
                    st.error("Unable to save account")
                    
            except Exception:
                st.error("Account creation failed")

# ========================
# LOGIN INTERFACE
# ========================
elif auth_choice == "Login":
    # Reload credentials
    credentials = load_credentials()
    authenticator.credentials = credentials
    
    num_users = len(credentials.get("usernames", {}))
    
    if num_users == 0:
        st.info("No accounts yet. Create one using 'Sign Up' in the sidebar")
        st.stop()
    
    # Check if logout was requested FIRST
    if 'logout_requested' in st.session_state and st.session_state['logout_requested']:
        # Clear everything
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Perform login
    try:
        name, authentication_status, username = authenticator.login('Login', 'main')
        
    except KeyError:
        st.warning("Session error. Please refresh the page")
        for key in ['name', 'authentication_status', 'username', 'logout']:
            if key in st.session_state:
                del st.session_state[key]
        st.stop()
        
    except Exception:
        authentication_status = None
        name = username = None
    
    # ========================
    # AUTHENTICATED USER - Main Application
    # ========================
    
    if authentication_status:
        # LOGOUT BUTTON
        if st.sidebar.button("🚪 Logout", key="logout_btn", type="primary"):
            st.session_state['logout_requested'] = True
            st.session_state['authentication_status'] = None
            st.session_state['name'] = None
            st.session_state['username'] = None
            st.rerun()
        
        st.markdown(f"Welcome back, **{name}**")
        st.markdown("---")
        
        # ========================
        # MODE SELECTOR - Single or Batch
        # ========================
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Single Review", "Batch Analysis"],
            horizontal=True
        )
        
        # ========================
        # SINGLE REVIEW ANALYSIS
        # ========================
        if analysis_mode == "Single Review":
            st.markdown("---")
            
            # Threshold slider
            threshold = st.slider(
                "Classification Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Adjust how strict the fake detection should be"
            )
            
            # Helper text for threshold
            if threshold == 0.5:
                st.caption("✓ Recommended: Keep at 0.5 for balanced results")
            elif threshold < 0.5:
                st.caption("⚠️ Lower threshold = More reviews marked as fake (more sensitive)")
            else:
                st.caption("⚠️ Higher threshold = Fewer reviews marked as fake (more strict)")
            
            review_input = st.text_area(
                "Review Text",
                height=150,
                placeholder="Paste the review you want to analyze here...",
                label_visibility="collapsed"
            )
            
            analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

            if analyze_btn:
                if not review_input or not review_input.strip():
                    st.warning("Please enter some review text first")
                    
                elif not HAS_MODEL:
                    st.info("Running in demo mode (model not available)")
                    
                else:
                    # Model loading (cached)
                    @st.cache_resource(show_spinner=False)
                    def _load_model_cached(path: str):
                        return load_model(path)

                    model_path = str(Path(MODELS_DIR) / "bert_fake_review_model.pkl")
                    
                    try:
                        with st.spinner("Analyzing review..."):
                            model = _load_model_cached(model_path)
                            
                            # Preprocessing
                            cleaned = clean_text(review_input)
                            emb = get_bert_embeddings([cleaned], max_len=128)
                            
                            if emb.ndim == 1:
                                emb = emb.reshape(1, -1)

                            # Classification
                            probs = model.predict_proba(emb)[0]
                            
                            # Label mapping
                            meta = read_metadata()
                            num_to_label = {0: "fake", 1: "genuine"}

                            # Probability calculation
                            class_prob = {int(c): float(p) for c, p in zip(model.classes_, probs)}
                            fake_num = next((n for n, lbl in num_to_label.items() if lbl == "fake"), 0)
                            p_fake = class_prob.get(fake_num, 0.0)

                            # Classification using user threshold
                            is_fake = p_fake >= threshold
                            
                            # Calculate confidence
                            confidence = p_fake if is_fake else (1 - p_fake)

                        # ========================
                        # RESULTS DISPLAY
                        # ========================
                        
                        st.markdown("---")
                        
                        if is_fake:
                            st.error("**Likely Fake Review**")
                            st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                            st.progress(confidence)
                            st.markdown("""
                            This review shows patterns commonly found in fake reviews. 
                            Exercise caution when relying on this review.
                            """)
                        else:
                            st.success("**Likely Genuine Review**")
                            st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                            st.progress(confidence)
                            st.markdown("""
                            This review appears authentic based on language patterns 
                            and writing style typical of real customer experiences.
                            """)
                        
                        # Details in collapsible section
                        with st.expander("Show Details"):
                            st.markdown("**Processed Text:**")
                            st.text(cleaned[:200] + "..." if len(cleaned) > 200 else cleaned)
                            
                            st.markdown("**Classification Probabilities:**")
                            st.json({
                                "Fake": f"{p_fake*100:.2f}%",
                                "Genuine": f"{(1-p_fake)*100:.2f}%"
                            })
                        
                        # Save to history
                        result_entry = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'review': review_input[:200],
                            'predicted_label': "fake" if is_fake else "genuine",
                            'p_fake': round(p_fake, 4),
                            'threshold_used': threshold,
                            'confidence': round(confidence * 100, 2)
                        }
                        st.session_state['single_review_history'].append(result_entry)
                        
                        # Download option
                        st.markdown("---")
                        if st.button("Save this result", use_container_width=True):
                            # Create single result CSV
                            result_df = pd.DataFrame([result_entry])
                            csv_buffer = BytesIO()
                            result_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                            csv_buffer.seek(0)
                            
                            st.download_button(
                                label="Download result as CSV",
                                data=csv_buffer.getvalue(),
                                file_name=f"review_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                    except Exception as e:
                        st.error("Analysis failed. Please try again")
                        with st.expander("Error details"):
                            st.code(str(e))
            
            # Show analysis history
            if len(st.session_state['single_review_history']) > 0:
                st.markdown("---")
                with st.expander(f"Analysis History ({len(st.session_state['single_review_history'])} results)"):
                    history_df = pd.DataFrame(st.session_state['single_review_history'])
                    st.dataframe(history_df, use_container_width=True)
                    
                    # Download all history
                    csv_buffer = BytesIO()
                    history_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                    csv_buffer.seek(0)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.download_button(
                            label="Download all history",
                            data=csv_buffer.getvalue(),
                            file_name=f"review_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col2:
                        if st.button("Clear history", use_container_width=True):
                            st.session_state['single_review_history'] = []
                            st.rerun()
        
        # ========================
        # BATCH ANALYSIS (CSV)
        # ========================
        else:  # Batch Analysis mode
            st.markdown("---")
            st.markdown("### Batch Review Analysis")
            st.markdown("Upload a CSV file containing reviews for bulk analysis")
            
            # Threshold for batch
            batch_threshold = st.slider(
                "Classification Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Adjust how strict the fake detection should be for batch processing",
                key="batch_threshold"
            )
            
            if batch_threshold == 0.5:
                st.caption("✓ Recommended: Keep at 0.5 for balanced results")
            elif batch_threshold < 0.5:
                st.caption("⚠️ Lower threshold = More reviews marked as fake")
            else:
                st.caption("⚠️ Higher threshold = Fewer reviews marked as fake")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                help="Upload a CSV file with a column containing review text"
            )
            
            if uploaded_file is not None:
                try:
                    # Read CSV
                    df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
                    
                    # Show preview
                    with st.expander("Data Preview (first 5 rows)"):
                        st.dataframe(df.head())
                    
                    # Detect review column
                    review_candidates = []
                    common_names = ['review', 'text', 'sentence', 'content', 'comment', 'feedback', 'review_text']
                    
                    # Normalize column names for matching
                    col_lower = {col.lower().strip(): col for col in df.columns}
                    
                    for name in common_names:
                        if name in col_lower:
                            review_candidates.append(col_lower[name])
                    
                    # Column selection
                    st.markdown("#### Select Review Column")
                    
                    if len(review_candidates) == 1:
                        selected_col = st.selectbox(
                            "Detected review column:",
                            review_candidates,
                            index=0
                        )
                    elif len(review_candidates) > 1:
                        selected_col = st.selectbox(
                            "Multiple candidates found - select one:",
                            review_candidates
                        )
                    else:
                        selected_col = st.selectbox(
                            "No auto-detected column - please select:",
                            df.columns.tolist()
                        )
                    
                    # Processing parameters
                    st.markdown("#### Processing Settings")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        max_rows = st.number_input(
                            "Max rows to process",
                            min_value=1,
                            max_value=len(df),
                            value=min(500, len(df)),
                            help="Limit processing to avoid memory issues"
                        )
                    
                    with col2:
                        chunk_size = st.number_input(
                            "Batch size",
                            min_value=8,
                            max_value=256,
                            value=64,
                            help="BERT embedding batch size"
                        )
                    
                    # Run analysis button
                    if st.button("Run Batch Analysis", type="primary", use_container_width=True):
                        if not HAS_MODEL:
                            st.error("Model not available. Cannot run batch analysis.")
                        else:
                            # Load model
                            @st.cache_resource(show_spinner=False)
                            def _load_model_cached(path: str):
                                return load_model(path)

                            model_path = str(Path(MODELS_DIR) / "bert_fake_review_model.pkl")
                            
                            try:
                                model = _load_model_cached(model_path)
                                meta = read_metadata()
                                num_to_label = {0: "fake", 1: "genuine"}
                                
                                # Prepare data
                                df_process = df[[selected_col]].head(max_rows).copy()
                                df_process = df_process.dropna(subset=[selected_col]).reset_index(drop=True)
                                
                                if df_process.empty:
                                    st.error("No valid reviews found after removing empty rows")
                                    st.stop()
                                
                                total_reviews = len(df_process)
                                st.info(f"Processing {total_reviews} reviews...")
                                
                                # Progress tracking
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Storage for results
                                results = []
                                
                                # Process in chunks
                                num_chunks = (total_reviews + chunk_size - 1) // chunk_size
                                
                                for chunk_idx in range(num_chunks):
                                    start_idx = chunk_idx * chunk_size
                                    end_idx = min((chunk_idx + 1) * chunk_size, total_reviews)
                                    
                                    chunk_reviews = df_process.iloc[start_idx:end_idx][selected_col].tolist()
                                    
                                    # Update status
                                    status_text.text(f"Processing chunk {chunk_idx+1}/{num_chunks} (reviews {start_idx+1}-{end_idx})...")
                                    
                                    # Clean text
                                    cleaned_reviews = [clean_text(str(rev)) for rev in chunk_reviews]
                                    
                                    # Get embeddings
                                    emb = get_bert_embeddings(cleaned_reviews, batch_size=chunk_size, max_len=128)
                                    
                                    if emb.ndim == 1:
                                        emb = emb.reshape(1, -1)
                                    
                                    # Predict
                                    probs = model.predict_proba(emb)
                                    
                                    # Process each review in chunk
                                    for i, (orig_rev, clean_rev, prob_vec) in enumerate(zip(
                                        chunk_reviews, cleaned_reviews, probs
                                    )):
                                        # Get class probabilities
                                        class_prob = {int(c): float(p) for c, p in zip(model.classes_, prob_vec)}
                                        
                                        # Find fake probability
                                        fake_num = next((n for n, lbl in num_to_label.items() if lbl == "fake"), 0)
                                        p_fake = class_prob.get(fake_num, 0.0)
                                        
                                        # Apply user threshold
                                        is_fake = p_fake >= batch_threshold
                                        pred_label = "fake" if is_fake else "genuine"
                                        
                                        # Confidence
                                        confidence_pct = (p_fake if is_fake else (1 - p_fake)) * 100
                                        
                                        results.append({
                                            'original_review': str(orig_rev)[:500],
                                            'cleaned': clean_rev[:500],
                                            'predicted_label': pred_label,
                                            'p_fake': round(p_fake, 4),
                                            'threshold_used': batch_threshold,
                                            'confidence_pct': round(confidence_pct, 2)
                                        })
                                    
                                    # Update progress
                                    progress = (end_idx / total_reviews)
                                    progress_bar.progress(progress)
                                
                                # Complete
                                status_text.text("Analysis complete!")
                                progress_bar.progress(1.0)
                                
                                # Create results dataframe
                                results_df = pd.DataFrame(results)
                                
                                # Display summary
                                st.markdown("---")
                                st.markdown("### Results Summary")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                fake_count = (results_df['predicted_label'] == 'fake').sum()
                                genuine_count = (results_df['predicted_label'] == 'genuine').sum()
                                
                                col1.metric("Total Analyzed", len(results_df))
                                col2.metric("Fake Reviews", fake_count, delta=f"{fake_count/len(results_df)*100:.1f}%")
                                col3.metric("Genuine Reviews", genuine_count, delta=f"{genuine_count/len(results_df)*100:.1f}%")
                                
                                # Show results table
                                st.markdown("### Detailed Results (first 100)")
                                display_df = results_df.head(100).copy()
                                st.dataframe(
                                    display_df,
                                    use_container_width=True,
                                    height=400
                                )
                                
                                # Download button
                                st.markdown("### Download Results")
                                
                                # Convert to CSV
                                csv_buffer = BytesIO()
                                results_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                                csv_buffer.seek(0)
                                
                                st.download_button(
                                    label="Download batch_results.csv",
                                    data=csv_buffer.getvalue(),
                                    file_name="batch_results.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                                # Additional stats
                                with st.expander("Additional Statistics"):
                                    st.markdown("**Confidence Distribution:**")
                                    st.write(results_df['confidence_pct'].describe())
                                    
                                    st.markdown("**Fake Probability Distribution:**")
                                    st.write(results_df['p_fake'].describe())
                                
                            except Exception as e:
                                st.error(f"Batch analysis failed: {str(e)}")
                                with st.expander("Error details"):
                                    st.code(str(e))
                                    import traceback
                                    st.code(traceback.format_exc())
                
                except Exception as e:
                    st.error(f"Failed to read CSV: {str(e)}")
                    st.info("Make sure your file is a valid CSV with proper encoding (UTF-8)")

    elif authentication_status is False:
        # Invalid credentials
        st.error("Incorrect username or password")
        
        with st.expander("Need help?"):
            st.markdown("**Tips:**")
            st.markdown("- Usernames and passwords are case-sensitive")
            st.markdown("- Make sure you created an account first")
            st.markdown(f"- {num_users} account(s) are registered")
    
    else:
        # No login attempt
        st.info("Please enter your credentials above to continue")

# ========================
# Footer
# ========================
st.markdown("---")
st.caption("AI-Powered Review Analysis • Research Project")