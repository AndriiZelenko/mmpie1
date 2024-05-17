#!/usr/bin/env bash
while getopts a:n:u:d: flag
do
    case "${flag}" in
        a) author=${OPTARG};;
        n) name=${OPTARG};;
        u) urlname=${OPTARG};;
        d) description=${OPTARG};;
    esac
done

echo "Author: $author";
echo "Project Name: $name";
echo "Project URL name: $urlname";
echo "Project Description: $description";

echo "Renaming project..."

original_author="AndriiZelenko"
original_name="mmpie1"
original_urlname="https://github.com/jeremywurbs/mmpie1/"
original_description="Awesome mmpie1 created by AndriiZelenko"

for filename in $(git ls-files)
do
    if [[ $filename == *"workflows"* ]]; then
        continue
    fi
    sed -i "s@$original_author@$author@g" $filename
    sed -i "s@${original_name^}@${name^}@g" $filename
    sed -i "s@${original_name^^}@${name^^}@g" $filename
    sed -i "s@$original_name@$name@g" $filename
    sed -i "s@$original_urlname@$urlname@g" $filename
    sed -i "s@$original_description@$description@g" $filename
    echo "Renamed $filename"
done

mv mmpie1 $name
mv _README.md README.md -f

# This command runs only once on GHA!
rm -rf .github/template.yml
