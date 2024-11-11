from functools import lru_cache
from io import BytesIO

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from PIL import Image, UnidentifiedImageError


FETCH_CACHE_SIZE = 10
HANDLE_CACHE_SIZE = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:45.0) Gecko/20100101 Firefox/45.0"


app = FastAPI()


def resize_to_keep_aspect_ratio(img: Image.Image, desired_width: int, desired_height: int) -> Image.Image:
    new_img = Image.new("RGBA", (desired_width, desired_height))
    if img.width >= img.height:
        k = desired_width / img.width
        img = img.resize((desired_width, int(img.height * k)), Image.Resampling.BICUBIC)
        new_img.paste(img, (0, (desired_height - img.height) // 2))
    else:
        k = desired_height / img.height
        img = img.resize((int(img.width * k), desired_height), Image.Resampling.BICUBIC)
        new_img.paste(img, ((desired_width - img.width) // 2, 0))
    return new_img


@lru_cache(maxsize=FETCH_CACHE_SIZE)
def fetch_picture(img_url: str) -> Image.Image:
    headers = {
        "User-Agent": USER_AGENT,
        "Content-Type": "image/*"
    }
    response = requests.get(img_url, headers=headers, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGBA")


@lru_cache(maxsize=HANDLE_CACHE_SIZE)
def handle_picture(img_url: str, width: int, height: int, keep_aspect_ratio: bool = False) -> np.ndarray:
    img = fetch_picture(img_url)
    img = (img.resize((width, height), Image.Resampling.BICUBIC) 
           if not keep_aspect_ratio 
           else resize_to_keep_aspect_ratio(img, width, height))
    arr = np.array(img.getdata(), dtype="uint8")
    new_arr = np.zeros(arr.shape[0], "uint32")
    for i, (r, g, b, a) in enumerate(arr):
        new_arr[i] = (int(a) << 24) | (int(r) << 16) | (int(g) << 8) | int(b)
    return new_arr


def error_message(message) -> str:
    return f"!{message}"


@app.get("/picpix", response_class=PlainTextResponse)
def get_picutre_pixels(img_url: str, i: int, cap: int, width: int, height: int, keep_aspect_ratio: bool = False):
    if i < 0:
        raise HTTPException(400, "Start index less than zero")
    if cap <= 0:
        raise HTTPException(400, "Buffer capacity is less or equal zero")
    if i >= width * height:
        raise HTTPException(400, "Start index is more than width * height")
    
    try:
        img_arr = handle_picture(img_url, width, height, keep_aspect_ratio)
    except requests.exceptions.HTTPError as errh:
        return error_message(f"HTTP error while requesting image url: {str(errh)}")
    except requests.exceptions.Timeout:
        return error_message(f"Timeout while requesting image url")
    except requests.exceptions.ConnectionError:
        return error_message(f"Connection error while requesting image url")
    except requests.exceptions.RequestException:
        return error_message(f"Unknown error while requesting image url")
    except UnidentifiedImageError as errimg:
        return error_message(f"Unidentified image: {str(errimg)}")
    
    buf = img_arr[i:(i + cap)]
    return " ".join(np.base_repr(pixel, 36) for pixel in buf)