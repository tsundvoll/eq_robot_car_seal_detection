import os

import pytest
from azure.cognitiveservices.vision.customvision.training.models import (
    Project,
    ProjectSettings,
    Tag,
)
from car_seal.custom_vision.upload_dataset import UploadDataset

mock_project = Project(
    name="mock_project", description="A mock project", settings=ProjectSettings()
)

mock_tag = Tag(name="mock_tag", description="Tag for test", type="For test")


@pytest.mark.parametrize(
    "annotation_path",
    [
        os.path.join(os.path.dirname(__file__), "test_dataset/1.txt"),
        os.path.join(os.path.dirname(__file__), "test_dataset/4.txt"),
        os.path.join(os.path.dirname(__file__), "test_dataset/34.txt"),
        os.path.join(os.path.dirname(__file__), "test_dataset/459.txt"),
        os.path.join(os.path.dirname(__file__), "test_dataset/580.txt"),
        os.path.join(os.path.dirname(__file__), "test_dataset/611.txt"),
        os.path.join(os.path.dirname(__file__), "test_dataset/1189.txt"),
    ],
)
def test_read_annotation_file(mocker, annotation_path):
    mocker.patch.object(
        UploadDataset, "_connect_to_or_create_project", return_value=mock_project
    )
    mocker.patch.object(UploadDataset, "_get_or_create_tag", return_value=mock_tag)
    upload_dataset = UploadDataset(files_to_upload=[], project_name="test")
    annotations: list = upload_dataset._read_annotation_file(
        annotation_path=annotation_path,
    )
    for annotation in annotations:
        assert annotation.left >= 0
        assert annotation.left + annotation.width <= 1
        assert annotation.top >= 0
        assert annotation.top + annotation.height <= 1
