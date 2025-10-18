import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.write("This app recommends movies using collaborative filtering (SVD) on the MovieLens dataset.")

# Cache data loading and model training
@st.cache_data
def load_data():
    """Load MovieLens dataset"""
    ratings = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', 
                          sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    movies = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.item',
                         sep='|', encoding='latin-1',
                         names=['movie_id', 'title', 'release_date', 'video_release_date', 
                                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 
                                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    return ratings, movies

@st.cache_resource
def train_model(ratings):
    """Train SVD model and return prediction matrix"""
    # Create user-item matrix
    user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
    
    # Fill NaN with 0
    user_item_matrix_filled = user_item_matrix.fillna(0)
    
    # Convert to sparse matrix
    sparse_matrix = csr_matrix(user_item_matrix_filled.values)
    
    # Apply SVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_features = svd.fit_transform(sparse_matrix)
    item_features = svd.components_
    
    # Reconstruct the matrix (predicted ratings)
    predicted_ratings = np.dot(user_features, item_features)
    
    # Create DataFrame for predictions
    predicted_df = pd.DataFrame(
        predicted_ratings,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )
    
    return predicted_df, user_item_matrix

# Load data
with st.spinner("Loading data and training model..."):
    ratings, movies = load_data()
    predicted_df, user_item_matrix = train_model(ratings)

st.success("✅ Model trained successfully!")

# Display dataset info
with st.expander("📊 Dataset Information"):
    st.write(f"**Total Ratings:** {len(ratings):,}")
    st.write(f"**Total Users:** {ratings['user_id'].nunique():,}")
    st.write(f"**Total Movies:** {ratings['movie_id'].nunique():,}")
    st.write(f"**Rating Scale:** 1-5 stars")

# User input
st.markdown("---")
st.subheader("🔍 Get Movie Recommendations")

user_id = st.number_input(
    "Enter User ID (1–943):", 
    min_value=1, 
    max_value=943, 
    value=100,
    help="Select a user ID to get personalized movie recommendations"
)

# Additional options
col1, col2 = st.columns(2)
with col1:
    n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
with col2:
    show_ratings = st.checkbox("Show predicted ratings", value=True)

if st.button("🎯 Get Recommendations", type="primary"):
    
    if user_id not in predicted_df.index:
        st.error(f"❌ User {user_id} not found in the dataset.")
    else:
        with st.spinner("Generating recommendations..."):
            # Get user's predictions
            user_predictions = predicted_df.loc[user_id]
            
            # Get movies user hasn't rated
            user_rated = user_item_matrix.loc[user_id]
            unrated_movies = user_rated[user_rated.isna()].index
            
            # Get predictions for unrated movies and sort
            recommendations = user_predictions[unrated_movies].sort_values(ascending=False)
            
            # Get top N
            top_recommendations = recommendations.head(n_recommendations)
            
            # Display recommendations
            st.markdown(f"### 🎬 Top {n_recommendations} Recommended Movies for User {user_id}:")
            
            # Create a nice display
            for idx, (movie_id, predicted_rating) in enumerate(top_recommendations.items(), 1):
                movie_info = movies[movies['movie_id'] == movie_id]
                if not movie_info.empty:
                    movie_title = movie_info['title'].values[0]
                    
                    # Get genres
                    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                                  'Thriller', 'War', 'Western']
                    
                    genres = [genre for genre in genre_cols if movie_info[genre].values[0] == 1]
                    genre_str = ", ".join(genres) if genres else "Unknown"
                    
                    # Display with nice formatting
                    if show_ratings:
                        st.markdown(f"""
                        **{idx}. {movie_title}**  
                        ⭐ Predicted Rating: `{predicted_rating:.2f}/5.00`  
                        🎭 Genres: {genre_str}
                        """)
                    else:
                        st.markdown(f"""
                        **{idx}. {movie_title}**  
                        🎭 Genres: {genre_str}
                        """)
            
            # Show what user has already rated
            st.markdown("---")
            with st.expander(f"📝 Movies User {user_id} Has Already Rated"):
                user_ratings = ratings[ratings['user_id'] == user_id].merge(
                    movies[['movie_id', 'title']], on='movie_id'
                )
                user_ratings_sorted = user_ratings.sort_values('rating', ascending=False)
                
                st.write(f"Total movies rated: {len(user_ratings)}")
                st.dataframe(
                    user_ratings_sorted[['title', 'rating']].head(20),
                    width='stretch',
                    hide_index=True
                )

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Dataset: MovieLens 100K • Model: SVD (Scikit-learn)")