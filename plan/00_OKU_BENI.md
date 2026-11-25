# Plan klasörü

Gökbilim ve astrofizik hesap kitaplığının (`kozmos`) planı. Hazırlanma tarihi 25 Kasım 2026. Kitaplığın ilk sürümü aynı gün yazılmıştır; uygulanan fonksiyonların güncel listesi [DURUM.md](DURUM.md) dosyasındadır.

## Okuma sırası

1. [01_vizyon_ve_kapsam.md](01_vizyon_ve_kapsam.md): amaç, kapsam, açık kararlar.
2. [02_saha_arastirmasi/](02_saha_arastirmasi/): var olan yazılım, veri ve güncel durum.
   - [01_yazilim_ekosistemi.md](02_saha_arastirmasi/01_yazilim_ekosistemi.md): alanlara göre kütüphaneler, sürümleri ve projedeki rolleri
   - [02_hubble_ve_kozmoloji_verisi.md](02_saha_arastirmasi/02_hubble_ve_kozmoloji_verisi.md): Pantheon+SH0ES veri sürümü, diğer veri setleri, `H0` ölçümlerinin durumu
   - [03_veri_arsivleri.md](02_saha_arastirmasi/03_veri_arsivleri.md): arşivler, erişim protokolleri, IVOA standartları
   - [04_arastirmanin_sinirlari.md](02_saha_arastirmasi/04_arastirmanin_sinirlari.md): hangi bilginin nasıl elde edildiği ve ne kadar güvenilir olduğu
3. [03_matematik/](03_matematik/)
   - [01_kuresel_astronomi_ve_donusumler.md](03_matematik/01_kuresel_astronomi_ve_donusumler.md)
   - [02_zaman_olcekleri.md](03_matematik/02_zaman_olcekleri.md)
   - [03_sayisal_yontemler.md](03_matematik/03_sayisal_yontemler.md)
   - [04_istatistik_ve_cikarim.md](03_matematik/04_istatistik_ve_cikarim.md)
   - [05_goruntu_ve_sinyal_isleme.md](03_matematik/05_goruntu_ve_sinyal_isleme.md)
4. [04_fizik/](04_fizik/)
   - [01_isinim_ve_fotometri.md](04_fizik/01_isinim_ve_fotometri.md)
   - [02_yorunge_ve_gok_mekanigi.md](04_fizik/02_yorunge_ve_gok_mekanigi.md)
   - [03_yildiz_astrofizigi.md](04_fizik/03_yildiz_astrofizigi.md)
   - [04_gokadalar_ve_dinamik.md](04_fizik/04_gokadalar_ve_dinamik.md)
   - [05_kozmoloji.md](04_fizik/05_kozmoloji.md)
   - [06_dedektor_ve_atmosfer.md](04_fizik/06_dedektor_ve_atmosfer.md)
   - [07_gorelilik_ve_yuksek_enerji.md](04_fizik/07_gorelilik_ve_yuksek_enerji.md)
   - [08_kara_delik_yildiz_ve_yildiz_olusumu.md](04_fizik/08_kara_delik_yildiz_ve_yildiz_olusumu.md): uygulanan üç fizik modülü ve kaynak okumasında bulunanlar
5. [05_veri_bilimi/](05_veri_bilimi/)
   - [01_veri_erisimi_ve_formatlar.md](05_veri_bilimi/01_veri_erisimi_ve_formatlar.md)
   - [02_goruntu_indirgeme_hatti.md](05_veri_bilimi/02_goruntu_indirgeme_hatti.md)
   - [03_fotometri_astrometri_tayf.md](05_veri_bilimi/03_fotometri_astrometri_tayf.md)
   - [04_zaman_serisi.md](05_veri_bilimi/04_zaman_serisi.md)
   - [05_veri_yonetimi_ve_tekrarlanabilirlik.md](05_veri_bilimi/05_veri_yonetimi_ve_tekrarlanabilirlik.md)
   - [06_optik_ve_kizilotesi_gozlem.md](05_veri_bilimi/06_optik_ve_kizilotesi_gozlem.md): uygulanan gözlem hattı, servislerin durumu, kızılötesi yapılacaklar
6. [06_mimari/](06_mimari/)
   - [01_paket_yapisi.md](06_mimari/01_paket_yapisi.md)
   - [02_kod_standartlari_ve_kaynak_gosterme.md](06_mimari/02_kod_standartlari_ve_kaynak_gosterme.md)
   - [03_test_ve_dogrulama.md](06_mimari/03_test_ve_dogrulama.md)
7. [07_yol_haritasi.md](07_yol_haritasi.md): aşamalar ve kabul ölçütleri.
8. [08_ornek_projeler/](08_ornek_projeler/)
   - [01_hubble_diyagrami.md](08_ornek_projeler/01_hubble_diyagrami.md)
   - [02_diger_projeler.md](08_ornek_projeler/02_diger_projeler.md)
9. [09_islev_envanteri.md](09_islev_envanteri.md): modül modül yazılacak fonksiyonlar, birimleri ve kaynakları.
10. [DURUM.md](DURUM.md): uygulanan fonksiyonlar, sabitler ve kaynak teyit durumu (kütükten üretilir).
11. [kaynakca/](../kaynakca/): `kaynaklar.bib`, `paketler.txt`, doğrulama betiği ve kaydı.

## Atıf kuralı

Köşeli ayraç içindeki anahtarlar (`[Hogg1999]`) `kaynakca/kaynaklar.bib` dosyasındaki girişlere işaret eder. Anahtarın ardından kaynağın içindeki yer gelebilir (`[Meeus1998, Böl. 7]`).

## Bilginin güven düzeyi

- Künyeler ve paket sürümleri dış kayıtlara karşı sınanmıştır; sonuç [kaynakca/DOGRULAMA_OZETI.md](../kaynakca/DOGRULAMA_OZETI.md) dosyasındadır.
- Bakım durumu, arşiv sürümü ve güncel ölçüm değerleri bir web taramasından gelir ve belgelerde işaretlidir.
- Plan belgelerindeki denklemler, katsayılar ve denklem numaraları hatırlanan bilgidir. Uygulanan fonksiyonlarda durum fonksiyon başına kayıtlıdır: `okundu` işaretli olanların denklemi kod yazılırken kaynağın metninden (arXiv TeX dosyaları, NIST sabit tablosu, ERFA kaynak kodu) alınmıştır; `bekliyor` işaretli olanların kaynağı henüz açılmamıştır.

Ayrıntı: [02_saha_arastirmasi/04_arastirmanin_sinirlari.md](02_saha_arastirmasi/04_arastirmanin_sinirlari.md).

## Doğrulamayı yeniden koşmak

```
python kaynakca/dogrula.py
```

Betik künyeleri Crossref, DataCite, arXiv ve Open Library kayıtlarıyla, paket sürümlerini PyPI ile karşılaştırır ve plan belgeleriyle kaynak kodundaki her atıf anahtarının kaynakçada tanımlı olduğunu denetler. Çıkış kodu sıfır değilse uyuşmayan künye ya da tanımsız anahtar vardır.
