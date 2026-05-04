
#
# Streamlit App: Pet Classifier, Similarity Search, and Painting Studio
#

import streamlit as st
from PIL import Image

from app_utils import (
    STYLE_LAYERS,
    CONTENT_LAYERS,
    STYLE_MAX_DIM_DEFAULT,
    TOP_K_DEFAULT,
    load_retrieval_metadata,
    load_embeddings,
    load_style_image_paths,
    load_tfds_splits,
    load_classifier_model,
    load_feature_extractor,
    infer_species_label_for_breed,
    retrieve_similar_images,
    retrieve_similar_images_from_embedding,
    get_display_image_by_index,
    get_content_tensor_from_dataset_index,
    pil_image_to_display_array,
    pil_image_to_content_tensor,
    classify_image,
    extract_uploaded_embedding,
    load_style_image_from_path,
    run_style_transfer
)


#
# Page Configuration
#

st.set_page_config(
    page_title="Pet Breed Classifier and Painting Studio",
    page_icon="🎨",
    layout="wide"
)


#
# App Title
#

st.title("Pet Breed Classifier, Similarity Search, and Painting Studio")

st.markdown(
    """
    This app demonstrates the Advanced Computer Vision group project.

    It combines:
    1. pet breed classification,
    2. similarity-based image retrieval,
    3. painting-style pet portrait generation.

    Note: the Oxford-IIIT Pet Dataset contains 37 cat and dog breeds. If an uploaded
    image is a breed outside the dataset, the app retrieves the closest available
    visual matches from the dataset rather than a guaranteed exact breed match.
    """
)


#
# Load Project Artifacts
#

try:
    retrieval_metadata_df = load_retrieval_metadata()
    normalized_embeddings = load_embeddings()
    style_image_paths = load_style_image_paths()
    ds_train_raw, ds_test_raw, ds_info, num_train, num_test = load_tfds_splits()

    breed_names = ds_info.features["label"].names

except Exception as exc:
    st.error(f"Could not load project artifacts: {exc}")
    st.stop()


#
# Sidebar Controls
#

st.sidebar.header("Image Source")

input_mode = st.sidebar.radio(
    "Choose image source",
    ["Select pet from dataset", "Upload my own pet image"]
)

top_k = st.sidebar.slider(
    "Number of similar pets to retrieve",
    min_value=1,
    max_value=10,
    value=TOP_K_DEFAULT
)

retrieval_filter = st.sidebar.radio(
    "Retrieval filter",
    ["Auto", "Dog only", "Cat only", "No filter"],
    index=0,
    help=(
        "Auto uses the selected pet's species or the uploaded image's predicted species. "
        "Dog only and Cat only are useful when an uploaded image is outside the dataset."
    )
)

st.sidebar.header("Painting Settings")

style_names = [
    path.stem.replace("_", " ").replace("-", " ").title()
    for path in style_image_paths
]

if len(style_names) == 0:
    st.sidebar.error("No style images found in assets/styles.")
    st.stop()

selected_style_name = st.sidebar.selectbox(
    "Choose painting style",
    style_names
)

selected_style_index = style_names.index(selected_style_name)
selected_style_path = style_image_paths[selected_style_index]

style_max_dim = st.sidebar.slider(
    "Painting image size",
    min_value=128,
    max_value=384,
    value=STYLE_MAX_DIM_DEFAULT,
    step=64
)

style_epochs = st.sidebar.slider(
    "Style transfer epochs",
    min_value=1,
    max_value=10,
    value=3
)

style_steps_per_epoch = st.sidebar.slider(
    "Steps per epoch",
    min_value=5,
    max_value=30,
    value=10,
    step=5
)

if st.sidebar.button("Refresh style list"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.image(
    Image.open(selected_style_path),
    caption=f"Selected style: {selected_style_path.name}",
    use_container_width=True
)


#
# Display Helper
#

def display_retrieved_images(results_df, title="Similar Pets"):
    """
    Display retrieved pet images in columns.
    """

    st.subheader(title)

    columns = st.columns(len(results_df))

    for column, (_, row) in zip(columns, results_df.iterrows()):
        retrieved_index = int(row["retrieved_index"])
        image = get_display_image_by_index(retrieved_index)

        caption = (
            f"Rank {int(row['retrieval_rank'])}\n\n"
            f"{row['breed_name']}\n\n"
            f"Similarity: {row['similarity_score']:.3f}"
        )

        column.image(image, caption=caption, use_container_width=True)


#
# Main App State
#

content_tensor_for_painting = None
retrieval_results_df = None
selected_pet_display = None
selected_pet_caption = None
auto_species_label = None


#
# Dataset Selection Mode
#

if input_mode == "Select pet from dataset":

    st.header("Select a Pet from the Dataset")

    breed_options = sorted(retrieval_metadata_df["breed_name"].unique().tolist())

    selected_breed = st.selectbox(
        "Choose a breed",
        breed_options
    )

    breed_subset_df = (
        retrieval_metadata_df
        .reset_index(names="query_index")
        .query("breed_name == @selected_breed")
    )

    query_options = breed_subset_df["query_index"].tolist()

    selected_query_index = st.selectbox(
        "Choose an image",
        query_options,
        format_func=lambda idx: f"{idx} - {retrieval_metadata_df.loc[idx, 'file_name']}"
    )

    query_row = retrieval_metadata_df.loc[selected_query_index]

    auto_species_label = int(query_row["species_label"])

    selected_pet_display = get_display_image_by_index(selected_query_index)
    selected_pet_caption = (
        f"{query_row['breed_name']} "
        f"({query_row['species_name']}) - {query_row['file_name']}"
    )

    st.subheader("Selected Pet")
    st.image(
        selected_pet_display,
        caption=selected_pet_caption,
        width=350
    )

    retrieval_results_df = retrieve_similar_images(
        query_index=selected_query_index,
        normalized_embeddings=normalized_embeddings,
        metadata_df=retrieval_metadata_df,
        top_k=top_k,
        retrieval_filter=retrieval_filter,
        auto_species_label=auto_species_label
    )

    display_retrieved_images(
        results_df=retrieval_results_df,
        title="Similar Pets from Dataset"
    )

    with st.expander("Show retrieval table"):
        st.dataframe(retrieval_results_df)

    content_tensor_for_painting = get_content_tensor_from_dataset_index(
        selected_query_index,
        max_dim=style_max_dim
    )


#
# Upload Mode
#

else:

    st.header("Upload Your Own Pet Image")

    uploaded_file = st.file_uploader(
        "Upload a pet image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        uploaded_image = Image.open(uploaded_file).convert("RGB")

        selected_pet_display = pil_image_to_display_array(uploaded_image)
        selected_pet_caption = "Uploaded pet image"

        st.subheader("Uploaded Pet")
        st.image(
            uploaded_image,
            caption="Uploaded pet image",
            width=350
        )

        try:
            classifier_model = load_classifier_model()
            feature_extractor = load_feature_extractor()

            predicted_index, predicted_breed, predicted_confidence, top_predictions_df = classify_image(
                pil_image=uploaded_image,
                classifier_model=classifier_model,
                breed_names=breed_names
            )

            auto_species_label = infer_species_label_for_breed(
                metadata_df=retrieval_metadata_df,
                breed_label=predicted_index
            )

            predicted_species_name = "Unknown"
            if auto_species_label is not None:
                predicted_species_name = (
                    retrieval_metadata_df
                    .loc[retrieval_metadata_df["species_label"] == auto_species_label, "species_name"]
                    .iloc[0]
                )

            st.subheader("Breed Prediction")
            st.write(f"**Predicted breed:** {predicted_breed}")
            st.write(f"**Predicted species for auto-filtering:** {predicted_species_name}")
            st.write(f"**Confidence:** {predicted_confidence:.2%}")

            with st.expander("Show top 5 predicted breeds"):
                st.dataframe(top_predictions_df)

            uploaded_embedding = extract_uploaded_embedding(
                pil_image=uploaded_image,
                feature_extractor=feature_extractor
            )

            retrieval_results_df = retrieve_similar_images_from_embedding(
                query_embedding=uploaded_embedding,
                normalized_embeddings=normalized_embeddings,
                metadata_df=retrieval_metadata_df,
                top_k=top_k,
                retrieval_filter=retrieval_filter,
                auto_species_label=auto_species_label
            )

            display_retrieved_images(
                results_df=retrieval_results_df,
                title="Similar Pets from Dataset"
            )

            with st.expander("Show retrieval table"):
                st.dataframe(retrieval_results_df)

            content_tensor_for_painting = pil_image_to_content_tensor(
                uploaded_image,
                max_dim=style_max_dim
            )

        except Exception as exc:
            st.error(f"Could not classify or retrieve similar pets: {exc}")


#
# Painting Generation
#

st.header("Generate a Painted Pet Portrait")

st.markdown(
    """
    The painting module uses the selected pet as the content image and the chosen
    style image as the artistic reference. Click the button below to generate the
    painted output.

    For faster results, use a smaller painting image size and fewer style-transfer
    steps. For better final presentation images, increase these settings.
    """
)

if content_tensor_for_painting is None:
    st.info("Select a dataset pet or upload a pet image before generating a painting.")
else:
    if st.button("Generate painted pet portrait"):

        with st.spinner("Generating painted pet portrait..."):

            style_tensor = load_style_image_from_path(
                selected_style_path,
                max_dim=style_max_dim
            )

            painted_image = run_style_transfer(
                content_image=content_tensor_for_painting,
                style_image=style_tensor,
                style_layers=STYLE_LAYERS,
                content_layers=CONTENT_LAYERS,
                epochs=style_epochs,
                steps_per_epoch=style_steps_per_epoch
            )

        st.subheader("Painted Pet Portrait")

        col1, col2, col3 = st.columns(3)

        col1.image(
            selected_pet_display,
            caption="Selected Pet",
            use_container_width=True
        )

        col2.image(
            Image.open(selected_style_path),
            caption=f"Style: {selected_style_name}",
            use_container_width=True
        )

        col3.image(
            painted_image,
            caption="Painted Pet Output",
            use_container_width=True
        )
