# app.py - Cross-Domain Fake Review Detection System
import streamlit as st
from pathlib import Path
import json
import streamlit_authenticator as stauth

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
# LOGIN INTERFACE - FIXED VERSION
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
        # MAIN REVIEW ANALYSIS INTERFACE
        # ========================
        
        review_input = st.text_area(
            "Review Text",
            height=150,
            placeholder="Paste the review you want to analyze here...",
            label_visibility="collapsed"
        )
        
        # Compact action row
        col1, col2 = st.columns([3, 1])
        with col1:
            domain = st.selectbox(
                "Category",
                ["app", "hotel", "yelp"],
                label_visibility="collapsed"
            )
        with col2:
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

                        # Classification threshold
                        thr = float(meta.get("thresholds", {}).get("default", 0.5))
                        is_fake = p_fake >= thr
                        
                        # Calculate confidence
                        confidence = p_fake if is_fake else (1 - p_fake)

                    # ========================
                    # RESULTS DISPLAY - Clean Color-Coded Cards
                    # ========================
                    
                    st.markdown("---")
                    
                    if is_fake:
                        # FAKE REVIEW - Red Warning
                        st.error("**Likely Fake Review**")
                        st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                        st.progress(confidence)
                        st.markdown("""
                        This review shows patterns commonly found in fake reviews. 
                        Exercise caution when relying on this review.
                        """)
                    else:
                        # GENUINE REVIEW - Green Success
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
                        
                except Exception as e:
                    st.error("Analysis failed. Please try again")
                    with st.expander("Need help?"):
                        st.write("If this problem continues, please contact support")

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