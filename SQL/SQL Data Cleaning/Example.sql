CREATE TABLE NashvilleHousing (
UniqueID bigint Primary Key, 
ParcelID VARCHAR,
LandUse VARCHAR, 
PropertyAddress VARCHAR,
SaleDate TIMESTAMP,
SalePrice VARCHAR,
LegalReference VARCHAR,
SoldAsVacant CHAR (10),
OwnerName VARCHAR,
OwnerAddress VARCHAR,
Acreage numeric,
TaxDistrict VARCHAR,
LandValue varchar,
BuildingValue varchar,
TotalValue varchar,
YearBuilt varchar,
Bedrooms varchar,
FullBath varchar,
HalfBath varchar
)


Select * from NashvilleHousing

--Standardize Date Format
Select saledate, CAST(saledate AS date)
from NashvilleHousing


ALTER TABLE NashvilleHousing
ADD SaleDateConverted Date

update NashvilleHousing
set SaleDateConverted= CAST(saledate AS date)

Select SaleDateConverted, CAST(saledate AS date)
from NashvilleHousing

--Populate Property Address data
Select *
from NashvilleHousing
--where propertyaddress is null
order by uniqueid

Select a.parcelid, a.propertyaddress, b.parcelid, b.propertyaddress, coalesce (a.propertyaddress, b.propertyaddress)
from NashvilleHousing as a
JOIN NashvilleHousing as b
	on a.parcelid=b.parcelid
	and a.uniqueid <>b.uniqueid
where a.propertyaddress is null

UPDATE NashvilleHousing as a
set propertyaddress= b.propertyaddress
from NashvilleHousing as b
WHERE 
	a.parcelid=b.parcelid
	and a.uniqueid <>b.uniqueid
	and a.propertyaddress is null

--Breaking out address into Individual columns (Address,City, States)

Select propertyaddress
from NashvilleHousing


Select
SUBSTRING (propertyaddress from 1 for position(',' in propertyaddress)-1) as address,
SUBSTRING (propertyaddress from (position(',' in propertyaddress)+1) for Length (propertyaddress)) as address
fROM NashvilleHousing


ALTER TABLE NashvilleHousing
ADD column propertysplitaddress VARCHAR 

update NashvilleHousing
set propertysplitaddress = SUBSTRING (propertyaddress from 1 for position(',' in propertyaddress)-1)

ALTER TABLE NashvilleHousing
ADD column propertysplitcity VARCHAR 

update NashvilleHousing
set propertysplitcity = SUBSTRING (propertyaddress from (position(',' in propertyaddress)+1) for Length (propertyaddress))

Select 
SPLIT_PART (owneraddress,',',1),
SPLIT_PART (owneraddress,',',2),
SPLIT_PART (owneraddress,',',3)
from NashvilleHousing

ALTER TABLE NashvilleHousing
ADD column ownersplitaddress VARCHAR 

update NashvilleHousing
set ownersplitaddress = SPLIT_PART (owneraddress,',',1)

ALTER TABLE NashvilleHousing
ADD column ownersplitcity VARCHAR 

update NashvilleHousing
set  ownersplitcity = SPLIT_PART (owneraddress,',',2)

ALTER TABLE NashvilleHousing
ADD column ownersplitstate VARCHAR 

update NashvilleHousing
set  ownersplitstate  = SPLIT_PART (owneraddress,',',3)

Select * from NashvilleHousing

--Change Y and N to Yes and No in 'Sold as Vacant' field

Select distinct (soldasvacant), Count(Soldasvacant)
from NashvilleHousing
Group by Soldasvacant
Order by 2

Select soldasvacant,
Case when soldasvacant ='Y' THEN 'Yes'
	when soldasvacant='N' THEN 'No'
	ELSE soldasvacant
	end 
from NashvilleHousing

UPDATE NashvilleHousing
SET soldasvacant = Case when soldasvacant ='Y' THEN 'Yes'
	when soldasvacant='N' THEN 'No'
	ELSE soldasvacant
	end 

--Remove Duplicate
 
With rownumCTE AS(
 Select *,
 	row_number () over(
	partition by parcelid,
				propertyaddress,
				saleprice,
				saledate,
				legalreference
				order by
				uniqueid) as row_num
 from	NashvilleHousing)
 
 Select * from rownumcte
 Where row_num = 1

-- Delete unused columns

Select * from NashvilleHousing

ALTER TABLE NashvilleHousing 
DROP COLUMN owneraddress,
DROP COLUMN propertyaddress,
DROP COLUMN taxdistrict 

ALTER TABLE NashvilleHousing
DROP COLUMN  Saledate
