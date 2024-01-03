n=$3
src=$1
des=$2
echo "repeat $src $n times to $des"

if [ $# -lt 3 ] || [ $# -gt 3 ]; then
  echo "usage: source_file destination_file repeat_times"
fi

if [ -f tmp ]; then
 rm tmp
fi
 
touch tmp
for i in `seq 1 $n`;do
  cat tmp $src > tmp1
  cp tmp1 tmp
done
mv tmp $des
rm tmp1
