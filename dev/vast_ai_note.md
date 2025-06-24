yeah there are some issues we had to handle - Unfortunately the large size is quite unavoidable, but the cache is disabled and we use copy-on-write instead of hardlinks because they cannot work with sync

Hopefully it will help, but a custom docker will always be the best way. Incidentally you do not need to change our dockerfile at all, simply start with

FROM vastai/base-image:<your preferred tag>

And then add a RUN to insert your own software. We will get a guide up for that asap