import logging
import numpy as np
from tinydb import TinyDB # type: ignore (tinydb type hint sorunları için)
from utils.helpers import handle_error
from common.constants import KNOWN_FACES_DB_PATH

class FaceStorage:
    """Bilinen yüzlerin descriptorlarını TinyDB kullanarak saklar ve yönetir."""
    def __init__(self, db_path=KNOWN_FACES_DB_PATH):
        self.db_path = db_path
        try:
            self.db = TinyDB(db_path, indent=4, ensure_ascii=False, encoding='utf-8')
        except Exception as e:
            handle_error(f"TinyDB başlatılırken hata ({db_path}): {e}", "DB Başlatma")
            raise # DB açılamazsa devam etmenin anlamı yok
        self.known_faces = {}
        self.load_known_faces()

    def load_known_faces(self):
        """Veritabanından bilinen yüzleri yükler."""
        self.known_faces = {}
        loaded_count = 0
        invalid_count = 0
        try:
            # Doğrudan tüm kayıtları almayı dene
            all_entries = self.db.all()
        except Exception as e:
            handle_error(f"Veritabanı okunurken hata ({self.db_path}): {e}", "DB Okuma")
            all_entries = [] # Hata varsa boş liste ile devam et

        data_to_process = []
        # TinyDB yapısını kontrol et
        if all_entries:
            if isinstance(all_entries, list) and len(all_entries) > 0:
                # Eğer liste içindeki ilk eleman '_default' anahtarlı bir dict ise (eski yapı?)
                if isinstance(all_entries[0], dict) and '_default' in all_entries[0]:
                    default_table = all_entries[0]['_default']
                    if isinstance(default_table, dict):
                        data_to_process = list(default_table.values())
                    else: # Beklenmedik yapı
                        handle_error(f"DBLoad: Beklenmedik '_default' tablo yapısı: {type(default_table)}", "DB Yapı")
                # Doğrudan kayıt listesi ise
                elif all(isinstance(item, dict) for item in all_entries):
                    data_to_process = all_entries
                else:
                    handle_error(f"DBLoad: Tanınmayan veritabanı yapısı: {all_entries}", "DB Yapı")
            elif isinstance(all_entries, list): # Boş liste ise
                 pass # Sorun yok
            else: # Liste değilse
                 handle_error(f"DBLoad: Veritabanı içeriği beklenmedik tipte: {type(all_entries)}", "DB Yapı")

        for record in data_to_process:
            if isinstance(record, dict) and 'name' in record and 'descriptor' in record:
                name = record['name']
                desc_list = record['descriptor']
                if isinstance(desc_list, list) and len(desc_list) == 128:
                    try:
                        np_desc = np.array(desc_list, dtype=np.float64)
                        if not np.isnan(np_desc).any() and not np.isinf(np_desc).any():
                            self.known_faces[name] = np_desc
                            loaded_count += 1
                        else:
                            handle_error(f"DBLoad: '{name}' NaN/Inf descriptor, atlandı.", "DB Yükleme")
                            invalid_count += 1
                    except Exception as array_err:
                        handle_error(f"DBLoad: '{name}' descriptor array hatası ({array_err}), atlandı.", "DB Yükleme")
                        invalid_count += 1
                else:
                    handle_error(f"DBLoad: '{name}' geçersiz descriptor formatı, atlandı.", "DB Yükleme")
                    invalid_count += 1
            else:
                handle_error(f"DBLoad: Geçersiz kayıt formatı, atlandı: {record}", "DB Yükleme")
                invalid_count += 1

        if loaded_count > 0:
            logging.info(f"{loaded_count} geçerli yüz kaydı yüklendi.")
        if invalid_count > 0:
            logging.warning(f"{invalid_count} geçersiz kayıt atlandı.")
        if not self.known_faces and invalid_count == 0:
            logging.warning("Veritabanı boş veya geçerli kayıt içermiyor.")

    def save_known_faces(self):
        """Hafızadaki bilinen yüzleri veritabanına kaydeder."""
        try:
            entries = [{'name': name, 'descriptor': desc.tolist()} for name, desc in self.known_faces.items()]
            self.db.truncate() # Önce mevcut veriyi temizle
            if entries:
                self.db.insert_multiple(entries) # Sonra yenilerini ekle
        except Exception as e:
            handle_error(f"Veritabanına kaydetme hatası ({self.db_path}): {e}", "DB Kaydetme")

    def add_face(self, name, descriptor):
        """Yeni bir yüzü ekler/günceller."""
        name = name.strip()
        if not name: 
            handle_error("Yüz eklenemedi: İsim boş.", "Yüz Ekleme")
            return False
        if descriptor is None or not isinstance(descriptor, np.ndarray) or descriptor.shape != (128,):
            handle_error(f"Yüz eklenemedi ('{name}'): Geçersiz descriptor.", "Yüz Ekleme")
            return False
        if np.isnan(descriptor).any() or np.isinf(descriptor).any():
            handle_error(f"Yüz eklenemedi ('{name}'): NaN/Inf descriptor.", "Yüz Ekleme")
            return False
        self.known_faces[name] = descriptor.astype(np.float64)
        self.save_known_faces()
        logging.info(f"'{name}' yüzü başarıyla eklendi/güncellendi.")
        return True

    def delete_face(self, name):
        """Verilen isimdeki yüzü siler."""
        name = name.strip()
        if name in self.known_faces:
            del self.known_faces[name]
            self.save_known_faces()
            logging.info(f"'{name}' yüzü başarıyla silindi.")
            return True
        handle_error(f"Silinecek yüz bulunamadı: '{name}'", "Yüz Silme")
        return False

    def get_known_faces(self):
        """Hafızadaki bilinen yüzler sözlüğünü döndürür."""
        return self.known_faces 