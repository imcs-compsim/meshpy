# The MeshPy website

The MeshPy website is based on the static site generator `jekyll`.

## Setup `jekyll`

Install the required packages (the given example is for Debian, the package names might vary for different distributions)
```bash
apt install ruby-dev ruby-bundler
```

## How to test changes to the website?

After making changes to the website, wo to the website folder and run
```bash
bundle exec jekyll serve
```
this will start a local server that you can use to inspect the changes to the website.
