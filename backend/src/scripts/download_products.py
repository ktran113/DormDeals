"""
Downloads products from Amazon Reviews 2023 dataset.
Filters using keywords and categories then downloads images and stores
data in SQLite.
"""

import os
import sys
import json
import requests
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).parent.parent))  #allow flat imports from src/

from database import engine, SessionLocal
from models import Base, Product

#keywords for college dorm products
KEYWORDS = [
    #furniture
    "desk", "chair",
    "bed", "bed frame", "loft bed", "bunk bed",
    "futon", "sofa", "sleeper sofa", "couch",
    "dresser", "drawer", "nightstand",
    "bookshelf", "shelf", "bookcase",
    "tv stand", "media console",
    "table", "dining table", "coffee table", "side table",
    
    #storage
    "storage", "storage bin", "bin", "basket", "tote",
    "organizer", "cube organizer", "cubby",
    "shoe rack", "shoe organizer",
    "cart", "rolling cart", "utility cart", "caddy",
    "hook", "hooks", "over the door",
    "hanger", "hangers",
    "laundry basket", "laundry hamper", "hamper",
    "trash can", "recycling bin",
    
    #bedding
    "mattress", "mattress topper", "topper",
    "pillow", "body pillow", "throw pillow",
    "blanket", "comforter", "duvet", "quilt",
    "bedding", "sheet", "sheet set",
    
    #decorations
    "rug", "carpet", "floor mat", "bath mat",
    "lamp", "desk lamp", "floor lamp", "table lamp",
    "light", "led strip", "string lights", "fairy lights",
    "mirror", "wall mirror",
    "curtain", "curtains", "blinds",
    "poster", "wall art", "tapestry", "frame",
    
    #kitchen
    "fridge", "mini fridge", "refrigerator",
    "microwave", "air fryer", "instant pot",
    "rice cooker", "slow cooker", "crock pot",
    "blender", "coffee maker", "keurig", "kettle",
    "toaster", "toaster oven",
    
    "dish rack", "cutting board",
    "tupperware", "container",
    "cooler",
    "plates", "bowl", "cookware",
    
    #cleaning / home util
    "vacuum", "handheld vacuum",
    "fan", "box fan", "desk fan", "tower fan",
    "broom", "dustpan",
    "heater", "space heater",
    "humidifier", "dehumidifier", "air purifier",
    
    #electronics
    "monitor", "keyboard",
    "mouse", "gaming mouse",
    "laptop", "macbook", "chromebook",
    "tablet", "ipad",
    "printer", "scanner",
    
    "headphone", "headphones", "earbuds", "airpods",
    "speaker", "bluetooth speaker", "soundbar",
    "webcam", "microphone",
    "tv", "television",
    
    "docking station", "usb-c hub",
    "charger", "charging cable",
    "cable", "hdmi cable", "usb cable",
    "adapter", "usb adapter", "dongle",
    "extension cord", "power strip", "surge protector",
    "laptop stand", "monitor stand",
    "mousepad", "desk pad",
    "phone stand", "tablet stand", "ring light",
    
    "gaming console", "console",
    "xbox", "playstation", "ps4", "ps5", "nintendo", "switch",
    "controller", "game controller",
    
    "roku", "firestick", "fire stick", "chromecast", "apple tv",
    
    #study tools
    "calculator",
    "whiteboard", "dry erase board", "corkboard", "bulletin board",
    "calendar", "planner", "notebook",
    "desk organizer", "pen holder", "filing cabinet", "binder",
    
    #bags
    "backpack", "laptop bag", "tote bag", "duffel bag",
    
    #misc
    "water bottle", "thermos",
    "mug",
    "clock", "alarm clock",
    
    # overall brands
    "ikea", "costco", "walmart", "amazon basics"
]

# Categories to search - using HuggingFace raw files
BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories"
#no "Furniture" category exists in this dataset (meta_Furniture.jsonl 404s);
#furniture items are reached via KEYWORDS against Home_and_Kitchen / Office_Products
CATEGORIES = [
    "Electronics",
    "Home_and_Kitchen",
    "Office_Products",
    "Appliances",
    "Video_Games",
    "Tools_and_Home_Improvement",
]

#target limit
TARGET_COUNT_CAT = 2500

# Cache directory for downloaded data
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


def matches_keywords(title: str) -> bool:
    #check title contains any keywords
    if not title:
        return False
    title_lower = title.lower()
    return any(kw in title_lower for kw in KEYWORDS)


def download_image(url: str, asin: str, retries: int = 3):
    if not url:
        return None

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            #determine extension from content type or URL
            ext = ".jpg"
            if "png" in url.lower() or "image/png" in response.headers.get("content-type", ""):
                ext = ".png"

            filepath = IMAGE_DIR / f"{asin}{ext}"
            filepath.write_bytes(response.content)
            return str(filepath)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)  # wait before retry
                continue
            print(f"  Failed to download image for {asin} after {retries} attempts: {e}")
            return None


def get_best_image(images: list):
    #takes best use image from best
    if not images:
        return None

    # images can be a list of dicts with 'hi_res', 'large', 'thumb' keys
    # or just a list of URLs
    for img in images:
        if isinstance(img, dict):
            #prefer hi_res, then large, then thumb
            for key in ["hi_res", "large", "thumb"]:
                if img.get(key):
                    return img[key]
        elif isinstance(img, str):
            return img
    return None


def parse_price(price_str):
    #find price
    if not price_str:
        return None
    try:
        cleaned = str(price_str).replace("$", "").replace(",", "").strip()
        return float(cleaned)
    except:
        return None


BATCH_SIZE = 100  # can adjust higher if needed
def process_batch(batch, db, category_name):
    # Processes a batch of candidates - stores metadata and image URLs only.
    # Images displayed via URL in frontend, fetched temporarily for embeddings.

    if not batch:
        return []

    # dedupe batch to avoid unique constraint errors
    seen = set()
    unique_batch = []
    for item in batch:
        if item["asin"] not in seen:
            seen.add(item["asin"])
            unique_batch.append(item)
    batch = unique_batch

    # extract ASINs and check which exist in one query
    batch_asins = [item["asin"] for item in batch]
    existing = set(
        asin for (asin,) in
        db.query(Product.asin).filter(Product.asin.in_(batch_asins)).all()
    )

    results = []
    for item in batch:
        if item["asin"] in existing:
            continue

        print(f"  Adding: {item['title'][:50]}...")

        #make record - store URL only, no local download
        product = Product(
            asin=item["asin"],
            title=item["title"],
            category=category_name,
            price=item["price"],
            image_url=item["image_url"],
            local_image_path=None  # not downloading, will use URL
        )
        results.append(product)

    #batch insert products
    if results:
        try:
            db.add_all(results)
            db.commit()
        except Exception as e:
            print(f"  Failed to commit batch: {e}")
            db.rollback()
            return []

    return results


def stream_jsonl(url):
    #stream and parse json from url
    print(f"  Streaming from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Parse line by line
    for line in response.iter_lines():
        if line:
            try:
                yield json.loads(line.decode('utf-8'))
            except json.JSONDecodeError:
                continue


def download_products():
    #download and store products

    #initalize tables
    Base.metadata.create_all(bind=engine)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    products_added = 0

    try:
        for category in CATEGORIES:
            print(f"\nLoading {category}")
            url = f"{BASE_URL}/meta_{category}.jsonl"

            # per-category cap counts what's already in the DB so re-runs keep
            # existing rows (e.g. the 2,500 Electronics) and only top up to the cap
            existing_count = db.query(Product).filter(Product.category == category).count()
            category_added = 0
            if existing_count >= TARGET_COUNT_CAT:
                print(f"  {category}: already has {existing_count}, skipping")
                continue

            try:
                batch = []
                for item in stream_jsonl(url):
                    if existing_count + category_added >= TARGET_COUNT_CAT:
                        break
                    title = item.get("title", "")

                    if not matches_keywords(title):
                        continue

                    asin = item.get("parent_asin") or item.get("asin")
                    if not asin:
                        continue

                    # Get image URL early to skip items without images
                    images = item.get("images", [])
                    image_url = get_best_image(images)
                    if not image_url:
                        continue

                    #adding to batch
                    batch.append({
                        "asin": asin,
                        "title": title,
                        "price": parse_price(item.get("price")),
                        "image_url": image_url,
                    })

                    #processing batch
                    if len(batch) >= BATCH_SIZE:
                        print(f"\n[{category}: {existing_count + category_added}/{TARGET_COUNT_CAT}] Processing batch of {len(batch)}...")
                        added = process_batch(batch, db, category)
                        category_added += len(added)
                        products_added += len(added)
                        batch = []

                        # Rate limit between batches
                        time.sleep(0.5)

                #process remaining
                if batch and existing_count + category_added < TARGET_COUNT_CAT:
                    print(f"\n[{category}: {existing_count + category_added}/{TARGET_COUNT_CAT}] Processing final batch of {len(batch)}...")
                    added = process_batch(batch, db, category)
                    category_added += len(added)
                    products_added += len(added)

            except Exception as e:
                print(f"  Error loading {category}: {e}")
                continue

            print(f"  {category}: added {category_added} (now {existing_count + category_added})")

        print(f"\nDownloaded {products_added} products this run")

    finally:
        db.close()


if __name__ == "__main__":
    download_products()
