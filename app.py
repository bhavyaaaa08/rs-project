import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

# Page configuration
st.set_page_config(
    page_title="E-commerce Recommender System",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        # Try to load from kagglehub
        import kagglehub
        path = kagglehub.dataset_download("vibivij/amazon-electronics-rating-datasetrecommendation")
        file_name = "ratings_Electronics.csv"
        df = pd.read_csv(os.path.join(path, file_name))
    except:
        st.error("⚠️ Please ensure the dataset is available. Run the data download scripts first.")
        return None, None, None
    
    df.columns = ['user_id', 'prod_id', 'rating', 'timestamp']
    df = df.drop('timestamp', axis=1)
    
    # Filter users with 50+ ratings
    counts = df['user_id'].value_counts()
    df_final = df[df['user_id'].isin(counts[counts >= 50].index)]
    
    # Create interaction matrix
    final_ratings_matrix = df_final.pivot(index='user_id', columns='prod_id', values='rating').fillna(0)
    final_ratings_matrix['user_index'] = np.arange(0, final_ratings_matrix.shape[0])
    final_ratings_matrix.set_index(['user_index'], inplace=True)
    
    return df_final, final_ratings_matrix, df

# Rank-based recommendation
def rank_based_recommendations(df_final, n=5, min_interaction=50):
    """Generate rank-based recommendations"""
    df_final['rating'] = pd.to_numeric(df_final['rating'], errors='coerce')
    average_rating = df_final.groupby('prod_id')['rating'].mean()
    count_rating = df_final.groupby('prod_id')['rating'].count()
    
    final_rating = pd.DataFrame({'avg_rating': average_rating, 'rating_count': count_rating})
    recommendations = final_rating[final_rating['rating_count'] > min_interaction]
    recommendations = recommendations.sort_values('avg_rating', ascending=False)
    
    return recommendations.head(n)

# User-based collaborative filtering
def similar_users(user_index, interactions_matrix):
    """Find similar users"""
    similarity = []
    for user in range(0, interactions_matrix.shape[0]):
        sim = cosine_similarity([interactions_matrix.loc[user_index]], [interactions_matrix.loc[user]])
        similarity.append((user, sim))
    
    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [tup[0] for tup in similarity]
    similarity_score = [tup[1] for tup in similarity]
    
    most_similar_users.remove(user_index)
    similarity_score.remove(similarity_score[0])
    
    return most_similar_users, similarity_score

def user_based_recommendations(user_index, num_of_products, interactions_matrix):
    """Generate user-based CF recommendations"""
    most_similar_users = similar_users(user_index, interactions_matrix)[0]
    
    prod_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[user_index] > 0)]))
    recommendations = []
    observed_interactions = prod_ids.copy()
    
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_products:
            similar_user_prod_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[similar_user] > 0)]))
            recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
            observed_interactions = observed_interactions.union(similar_user_prod_ids)
        else:
            break
    
    return recommendations[:num_of_products]

# Model-based collaborative filtering
@st.cache_data
def train_svd_model(_final_ratings_matrix):
    """Train SVD model"""
    final_ratings_sparse = csr_matrix(_final_ratings_matrix.values)
    U, s, Vt = svds(final_ratings_sparse, k=50)
    sigma = np.diag(s)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    preds_df = pd.DataFrame(abs(all_user_predicted_ratings), columns=_final_ratings_matrix.columns)
    preds_matrix = csr_matrix(preds_df.values)
    return final_ratings_sparse, preds_matrix

def model_based_recommendations(user_index, interactions_matrix, preds_matrix, num_recommendations):
    """Generate model-based recommendations"""
    user_ratings = interactions_matrix[user_index,:].toarray().reshape(-1)
    user_predictions = preds_matrix[user_index,:].toarray().reshape(-1)
    
    temp = pd.DataFrame({'user_ratings': user_ratings, 'user_predictions': user_predictions})
    temp['product_id'] = np.arange(len(user_ratings))
    temp = temp.loc[temp.user_ratings == 0]
    temp = temp.sort_values('user_predictions', ascending=False)
    
    return temp.head(num_recommendations)

# Main app
def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">🛒 E-commerce Recommender System</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0;">Amazon Electronics Product Recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        df_final, final_ratings_matrix, df = load_data()
    
    if df_final is None:
        return
    
    # Sidebar
    st.sidebar.header("📊 Dataset Statistics")
    st.sidebar.metric("Total Users", df_final['user_id'].nunique())
    st.sidebar.metric("Total Products", df_final['prod_id'].nunique())
    st.sidebar.metric("Total Ratings", len(df_final))
    
    density = (np.count_nonzero(final_ratings_matrix) / (final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1])) * 100
    st.sidebar.metric("Matrix Density", f"{density:.2f}%")
    
    # Main content
    st.header("🎯 Choose Recommendation System")
    
    # System selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rank_selected = st.button("📈 Rank-Based", use_container_width=True)
    with col2:
        user_selected = st.button("👥 User-Based CF", use_container_width=True)
    with col3:
        model_selected = st.button("✨ Model-Based (SVD)", use_container_width=True)
    
    # Store selection in session state
    if rank_selected:
        st.session_state['system'] = 'rank'
    elif user_selected:
        st.session_state['system'] = 'user'
    elif model_selected:
        st.session_state['system'] = 'model'
    
    if 'system' not in st.session_state:
        st.session_state['system'] = 'rank'
    
    selected_system = st.session_state['system']
    
    # Display system info
    st.markdown("---")
    
    system_info = {
        'rank': {
            'title': '📈 Rank-Based Recommendations',
            'desc': 'Recommends popular products based on average ratings and interaction count.',
            'features': ['✅ Best for new users', '✅ No cold start problem', '✅ Based on overall popularity']
        },
        'user': {
            'title': '👥 User-Based Collaborative Filtering',
            'desc': 'Finds similar users and recommends products they liked.',
            'features': ['✅ Personalized recommendations', '✅ Uses cosine similarity', '✅ Based on user behavior']
        },
        'model': {
            'title': '✨ Model-Based Collaborative Filtering (SVD)',
            'desc': 'Uses matrix factorization to predict ratings for unseen products.',
            'features': ['✅ High accuracy', '✅ Handles sparse data', '✅ 50 latent features']
        }
    }
    
    info = system_info[selected_system]
    st.subheader(info['title'])
    st.info(info['desc'])
    
    col1, col2, col3 = st.columns(3)
    for i, feature in enumerate(info['features']):
        with [col1, col2, col3][i]:
            st.markdown(f"**{feature}**")
    
    st.markdown("---")
    
    # Configuration
    st.header("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if selected_system != 'rank':
            user_idx = st.number_input(
                "User Index",
                min_value=0,
                max_value=final_ratings_matrix.shape[0]-1,
                value=3,
                help=f"Enter user index (0 to {final_ratings_matrix.shape[0]-1})"
            )
        
        num_recommendations = st.number_input(
            "Number of Recommendations",
            min_value=1,
            max_value=20,
            value=5
        )
    
    with col2:
        if selected_system == 'rank':
            min_interactions = st.number_input(
                "Minimum Interactions",
                min_value=10,
                max_value=200,
                value=50,
                help="Minimum number of ratings a product must have"
            )
    
    # Generate recommendations button
    if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
        with st.spinner('Generating recommendations...'):
            try:
                if selected_system == 'rank':
                    recommendations = rank_based_recommendations(df_final, num_recommendations, min_interactions)
                    
                    st.success(f"✅ Generated {len(recommendations)} recommendations!")
                    st.markdown("---")
                    st.subheader("🏆 Top Recommended Products")
                    
                    for idx, (prod_id, row) in enumerate(recommendations.iterrows(), 1):
                        col1, col2, col3, col4 = st.columns([1, 4, 2, 2])
                        with col1:
                            st.markdown(f"### {idx}")
                        with col2:
                            st.markdown(f"**Product ID:** `{prod_id}`")
                        with col3:
                            st.metric("Avg Rating", f"{row['avg_rating']:.2f}")
                        with col4:
                            st.metric("Ratings Count", int(row['rating_count']))
                
                elif selected_system == 'user':
                    recommendations = user_based_recommendations(user_idx, num_recommendations, final_ratings_matrix)
                    
                    st.success(f"✅ Generated {len(recommendations)} recommendations for User {user_idx}!")
                    st.markdown("---")
                    st.subheader("🎯 Personalized Recommendations")
                    
                    for idx, prod_id in enumerate(recommendations, 1):
                        col1, col2 = st.columns([1, 5])
                        with col1:
                            st.markdown(f"### {idx}")
                        with col2:
                            st.markdown(f"**Product ID:** `{prod_id}`")
                
                elif selected_system == 'model':
                    interactions_matrix, preds_matrix = train_svd_model(final_ratings_matrix)
                    recommendations = model_based_recommendations(user_idx, interactions_matrix, preds_matrix, num_recommendations)
                    
                    st.success(f"✅ Generated {len(recommendations)} recommendations for User {user_idx}!")
                    st.markdown("---")
                    st.subheader("🤖 AI-Powered Recommendations")
                    
                    for idx, row in recommendations.iterrows():
                        col1, col2, col3 = st.columns([1, 4, 2])
                        with col1:
                            st.markdown(f"### {idx+1}")
                        with col2:
                            prod_id = final_ratings_matrix.columns[int(row['product_id'])]
                            st.markdown(f"**Product ID:** `{prod_id}`")
                        with col3:
                            st.metric("Predicted Rating", f"{row['user_predictions']:.2f}")
                
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>Built with Streamlit • Amazon Electronics Dataset</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()