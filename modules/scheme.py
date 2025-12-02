from pydantic import BaseModel, field_validator, model_validator
from fastapi import HTTPException, status
from numpy import array, isnan


class KNearestNeighborsIn(BaseModel):
    model_config = {
        'extra': 'forbid'
    }

    X: list[list[float]]
    y: list[int]


    @field_validator('X')
    def verify_X(cls, value):
        X = array(value)
        if X.ndim != 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='Must be a 2D matrix'
            )
        if isnan(X).any():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='Matrix X must not contain NaN values'
            )
        return X


    @field_validator('y')
    def verify_y(cls, value):
        y = array(value)
        if y.ndim != 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='Must be a 1D vector'
            )
        if isnan(y).any():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='Vector y must not contain NaN values'
            )
        return y

    @model_validator(mode='after')
    def verify_object(self):
        if self.X.shape[0] != self.y.shape[0]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='The number of n_sample and target values must be equal.'
            )
        return self


class KNearestNeighbors(BaseModel):
    model_config = {
        'extra': 'forbid'
    }

    X: list[float]

    @field_validator('X')
    def verify_X(cls, value):
        X = array(value)
        if X.ndim != 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='Must be a 1D vector'
            )
        if isnan(X).any():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail='Vector X must not contain NaN values'
            )
        return X