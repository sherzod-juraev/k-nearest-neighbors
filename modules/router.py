from fastapi import APIRouter, status
from .scheme import KNearestNeighborsPredict, KNearestNeighborsIn
from k_nearest_neighbors import KNearestNeighbors

modules_router = APIRouter()

k_nearest_neighbors = KNearestNeighbors()


@modules_router.post(
    '/',
    summary='K nearest neighbors model fit',
    status_code=status.HTTP_200_OK
)
async def model_fit(
        k_nearest_neighbors_scheme: KNearestNeighborsIn
):
    k_nearest_neighbors.fit(k_nearest_neighbors_scheme.X, k_nearest_neighbors_scheme.y)


@modules_router.post(
    '/predict',
    summary='Model predict',
    status_code=status.HTTP_200_OK
)
async def model_predict(
        k_nearest_neighbors_scheme: KNearestNeighborsPredict
):
    result = k_nearest_neighbors.predict(k_nearest_neighbors_scheme.X)
    return result